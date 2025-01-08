#include "openmc/cross_sections.h"

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/container_util.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/geometry_aux.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/photon.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/string_utils.h"
#include "openmc/thermal.h"
#include "openmc/timer.h"
#include "openmc/wmp.h"
#include "openmc/xml_interface.h"

#include "pugixml.hpp"

#include <cstdlib> // for getenv
#include <unordered_set>
#include <fmt/core.h>

namespace openmc {

//==============================================================================
// Global variable declarations
//==============================================================================

namespace data {

std::map<LibraryKey, std::size_t> library_map;
vector<Library> libraries;
} // namespace data

//==============================================================================
// Library methods
//==============================================================================

Library::Library(pugi::xml_node node, const std::string& directory)
{
  // Get type of library
  if (check_for_node(node, "type")) {
    auto type = get_node_value(node, "type");
    if (type == "neutron") {
      type_ = Type::neutron;
    } else if (type == "thermal") {
      type_ = Type::thermal;
    } else if (type == "photon") {
      type_ = Type::photon;
    } else if (type == "wmp") {
      type_ = Type::wmp;
    } else {
      fatal_error("Unrecognized library type: " + type);
    }
  } else {
    fatal_error("Missing library type");
  }

  // Get list of materials
  if (check_for_node(node, "materials")) {
    materials_ = get_node_array<std::string>(node, "materials");
  }

  // determine path of cross section table
  if (!check_for_node(node, "path")) {
    fatal_error("Missing library path");
  }
  std::string path = get_node_value(node, "path");

  if (starts_with(path, "/")) {
    path_ = path;
  } else if (ends_with(directory, "/")) {
    path_ = directory + path;
  } else if (!directory.empty()) {
    path_ = directory + "/" + path;
  } else {
    path_ = path;
  }

  if (!file_exists(path_)) {
    warning("Cross section library " + path_ + " does not exist.");
  }
}

//==============================================================================
// Non-member functions
//==============================================================================

void read_cross_sections_xml()
{
  pugi::xml_document doc;
  std::string filename = settings::path_input + "materials.xml";
  // Check if materials.xml exists
  if (!file_exists(filename)) {
    fatal_error("Material XML file '" + filename + "' does not exist.");
  }
  // Parse materials.xml file
  doc.load_file(filename.c_str());

  auto root = doc.document_element();

  read_cross_sections_xml(root);
}

void read_cross_sections_xml(pugi::xml_node root)
{
  // Find cross_sections.xml file -- the first place to look is the
  // materials.xml file. If no file is found there, then we check the
  // OPENMC_CROSS_SECTIONS environment variable
  if (!check_for_node(root, "cross_sections")) {
    // No cross_sections.xml file specified in settings.xml, check
    // environment variable
    if (settings::run_CE) {
      char* envvar = std::getenv("OPENMC_CROSS_SECTIONS");
      if (!envvar) {
        fatal_error(
          "No cross_sections.xml file was specified in "
          "materials.xml or in the OPENMC_CROSS_SECTIONS"
          " environment variable. OpenMC needs such a file to identify "
          "where to find data libraries. Please consult the"
          " user's guide at https://docs.openmc.org/ for "
          "information on how to set up data libraries.");
      }
      settings::path_cross_sections = envvar;
    } else {
      char* envvar = std::getenv("OPENMC_MG_CROSS_SECTIONS");
      if (!envvar) {
        fatal_error(
          "No mgxs.h5 file was specified in "
          "materials.xml or in the OPENMC_MG_CROSS_SECTIONS environment "
          "variable. OpenMC needs such a file to identify where to "
          "find MG cross section libraries. Please consult the user's "
          "guide at https://docs.openmc.org for information on "
          "how to set up MG cross section libraries.");
      }
      settings::path_cross_sections = envvar;
    }
  } else {
    settings::path_cross_sections = get_node_value(root, "cross_sections");

    // If no '/' found, the file is probably in the input directory
    auto pos = settings::path_cross_sections.rfind("/");
    if (pos == std::string::npos && !settings::path_input.empty()) {
      settings::path_cross_sections =
        settings::path_input + "/" + settings::path_cross_sections;
    }
  }

  // Now that the cross_sections.xml or mgxs.h5 has been located, read it in
  if (settings::run_CE) {
    read_ce_cross_sections_xml();
  } else {
    data::mg.read_header(settings::path_cross_sections);
    put_mgxs_header_data_to_globals();
  }

  // Establish mapping between (type, material) and index in libraries
  int i = 0;
  for (const auto& lib : data::libraries) {
    for (const auto& name : lib.materials_) {
      LibraryKey key {lib.type_, name};
      data::library_map.insert({key, i});
    }
    ++i;
  }

  // Check that 0K nuclides are listed in the cross_sections.xml file
  for (const auto& name : settings::res_scat_nuclides) {
    LibraryKey key {Library::Type::neutron, name};
    if (data::library_map.find(key) == data::library_map.end()) {
      fatal_error("Could not find resonant scatterer " + name +
                  " in cross_sections.xml file!");
    }
  }
}

void read_ce_cross_sections(const vector<vector<double>>& nuc_temps,
  const vector<vector<double>>& thermal_temps)
{
  std::unordered_set<std::string> already_read;

  // Construct a vector of nuclide names because we haven't loaded nuclide data
  // yet, but we need to know the name of the i-th nuclide
  vector<std::string> nuclide_names(data::nuclide_map.size());
  vector<std::string> thermal_names(data::thermal_scatt_map.size());
  for (const auto& kv : data::nuclide_map) {
    nuclide_names[kv.second] = kv.first;
  }
  for (const auto& kv : data::thermal_scatt_map) {
    thermal_names[kv.second] = kv.first;
  }

  // Read cross sections
  for (const auto& mat : model::materials) {
    for (int i_nuc : mat->nuclide_) {
      // Find name of corresponding nuclide. Because we haven't actually loaded
      // data, we don't have the name available, so instead we search through
      // all key/value pairs in nuclide_map
      std::string& name = nuclide_names[i_nuc];

      // If we've already read this nuclide, skip it
      if (already_read.find(name) != already_read.end())
        continue;

      const auto& temps = nuc_temps[i_nuc];
      int err = openmc_load_nuclide(name.c_str(), temps.data(), temps.size());
      if (err < 0)
        throw std::runtime_error {openmc_err_msg};

      already_read.insert(name);
    }
  }

  // Perform final tasks -- reading S(a,b) tables, normalizing densities
  for (auto& mat : model::materials) {
    for (const auto& table : mat->thermal_tables_) {
      // Get name of S(a,b) table
      int i_table = table.index_table;
      std::string& name = thermal_names[i_table];

      if (already_read.find(name) == already_read.end()) {
        LibraryKey key {Library::Type::thermal, name};
        int idx = data::library_map[key];
        std::string& filename = data::libraries[idx].path_;

        write_message(6, "Reading {} from {}", name, filename);

        // Open file and make sure version matches
        hid_t file_id = file_open(filename, 'r');
        check_data_version(file_id);

        // Read thermal scattering data from HDF5
        hid_t group = open_group(file_id, name.c_str());
        data::thermal_scatt.push_back(
          make_unique<ThermalScattering>(group, thermal_temps[i_table]));
        close_group(group);
        file_close(file_id);

        // Add name to dictionary
        already_read.insert(name);
      }
    } // thermal_tables_

    // Finish setting up materials (normalizing densities, etc.)
    mat->finalize();
  } // materials

  if (settings::photon_transport &&
      settings::electron_treatment == ElectronTreatment::TTB) {
    // Take logarithm of energies since they are log-log interpolated
    data::ttb_e_grid = xt::log(data::ttb_e_grid);
  }

  // Show minimum/maximum temperature
  write_message(
    4, "Minimum neutron data temperature: {} K", data::temperature_min);
  write_message(
    4, "Maximum neutron data temperature: {} K", data::temperature_max);

  // If the user wants multipole, make sure we found a multipole library.
  if (settings::temperature_multipole) {
    bool mp_found = false;
    for (const auto& nuc : data::nuclides) {
      if (nuc->multipole_) {
        mp_found = true;
        break;
      }
    }
    if (mpi::master && !mp_found) {
      warning("Windowed multipole functionality is turned on, but no multipole "
              "libraries were found. Make sure that windowed multipole data is "
              "present in your cross_sections.xml file.");
    }
  }
}

void read_ce_cross_sections_xml()
{
  // Check if cross_sections.xml exists
  const auto& filename = settings::path_cross_sections;
  if (dir_exists(filename)) {
    fatal_error("OPENMC_CROSS_SECTIONS is set to a directory. "
                "It should be set to an XML file.");
  }
  if (!file_exists(filename)) {
    // Could not find cross_sections.xml file
    fatal_error("Cross sections XML file '" + filename + "' does not exist.");
  }

  write_message("Reading cross sections XML file...", 5);

  // Parse cross_sections.xml file
  pugi::xml_document doc;
  auto result = doc.load_file(filename.c_str());
  if (!result) {
    fatal_error("Error processing cross_sections.xml file.");
  }
  auto root = doc.document_element();

  std::string directory;
  if (check_for_node(root, "directory")) {
    // Copy directory information if present
    directory = get_node_value(root, "directory");
  } else {
    // If no directory is listed in cross_sections.xml, by default select the
    // directory in which the cross_sections.xml file resides

    // TODO: Use std::filesystem functionality when C++17 is adopted
    auto pos = filename.rfind("/");
    if (pos == std::string::npos) {
      // No '\\' found, so the file must be in the same directory as
      // materials.xml
      directory = settings::path_input;
    } else {
      directory = filename.substr(0, pos);
    }
  }

  for (const auto& node_library : root.children("library")) {
    data::libraries.emplace_back(node_library, directory);
  }

  // Make sure file was not empty
  if (data::libraries.empty()) {
    fatal_error(
      "No cross section libraries present in cross_sections.xml file.");
  }
}

void finalize_cross_sections()
{
  if (settings::run_mode != RunMode::PLOTTING) {
    simulation::time_read_xs.start();
    if (settings::run_CE) {
      // Determine desired temperatures for each nuclide and S(a,b) table
      double_2dvec nuc_temps(data::nuclide_map.size());
      double_2dvec thermal_temps(data::thermal_scatt_map.size());
      get_temperatures(nuc_temps, thermal_temps);

      // Read continuous-energy cross sections from HDF5
      read_ce_cross_sections(nuc_temps, thermal_temps);
    } else {
      // Create material macroscopic data for MGXS
      set_mg_interface_nuclides_and_temps();
      data::mg.init();
      mark_fissionable_mgxs_materials();
    }
    simulation::time_read_xs.stop();
  }
}

// Friend function definition
void access_xs_types(const Nuclide& nuc, int& XS_TOTAL, int& XS_ABSORPTION, int& XS_FISSION, int& XS_NU_FISSION, int& XS_PHOTON_PROD) {
  XS_TOTAL = Nuclide::XS_TOTAL;
  XS_ABSORPTION = Nuclide::XS_ABSORPTION;
  XS_FISSION = Nuclide::XS_FISSION;
  XS_NU_FISSION = Nuclide::XS_NU_FISSION;
  XS_PHOTON_PROD = Nuclide::XS_PHOTON_PROD;
}


void randomly_sample_cross_sections()
{
  // Modify continuous-energy cross sections
  for (auto& mat : model::materials) {
  // Loop over each nuclide in the material
    for (int i_nuc : mat->nuclide_) {
      // Access the corresponding Nuclide object
      auto& nuc = data::nuclides[i_nuc];

      std::string nuclide_name = nuc->name_;

      // Access cross section type constants
      int XS_TOTAL, XS_ABSORPTION, XS_FISSION, XS_NU_FISSION, XS_PHOTON_PROD;
      access_xs_types(*nuc, XS_TOTAL, XS_ABSORPTION, XS_FISSION, XS_NU_FISSION, XS_PHOTON_PROD);
      
      /*fmt::print("Contents of random sample XS:\n");
      for (const auto& entry : settings::random_sample_xs) {
        fmt::print("Nuclide: {}\n", entry.first);
        fmt::print("Cross sections: ");
        for (const auto& xs_type : entry.second) {
          fmt::print("{} ", xs_type);
        }
        fmt::print("\n");
      }*/
      // Check if this nuclide is specified for random sampling
      if (settings::random_sample_xs.find(nuclide_name) != settings::random_sample_xs.end()) {

        //fmt::print("Modifying cross sections for nuclide: {}\n", nuclide_name);
        const auto& xs_types = settings::random_sample_xs[nuclide_name];

        // Generate a new seed based on the current batch number
        uint64_t seed = init_seed(simulation::current_batch, i_nuc);

        // Randomly sample the specified cross sections
        for (const auto& xs_type : xs_types) {
  
          if (xs_type == "total") {
            //int count = 0;
            // loop for different temperatures
            for (auto& xs : nuc->xs_) {
              // loop for different cross section values
              for (auto& value : xs)
              {
                value *= 1.0 + (prn(&seed) - 0.5) * 0.2; // Example perturbation
                //count++;
              }
            }
            //fmt::print("Number of elements in total cross section: {}\n", count);
          } 
          /*else if (xs_type == "fission") {
            int count = 0;
            for (auto& value : nuc->xs_[XS_FISSION]) {
              value *= 1.0 + (prn(&seed) - 0.5) * 0.1; // Example perturbation
              count++;
            }
            fmt::print("Number of elements in fission cross section: {}\n", count);
          } else if (xs_type == "absorption") {
            int count = 0;
            for (auto& value : nuc->xs_[XS_ABSORPTION]) {
              value *= 1.0 + (prn(&seed) - 0.5) * 0.1; // Example perturbation
              count++;
            }
            fmt::print("Number of elements in absorption cross section: {}\n", count);
          } else if (xs_type == "nu-fission") {
            int count = 0;
            for (auto& value : nuc->xs_[XS_NU_FISSION]) {
              value *= 1.0 + (prn(&seed) - 0.5) * 0.1; // Example perturbation
              count++;
            }
            fmt::print("Number of elements in nu-fission cross section: {}\n", count);
          } else if (xs_type == "photon-production") {
            int count = 0;
            for (auto& value : nuc->xs_[XS_PHOTON_PROD]) {
              value *= 1.0 + (prn(&seed) - 0.5) * 0.1; // Example perturbation
              count++;
            }
            fmt::print("Number of elements in photon-prod cross section: {}\n", count);
          }*/
        }
      }
    }
  }
}

void library_clear()
{
  data::libraries.clear();
  data::library_map.clear();
}

} // namespace openmc
