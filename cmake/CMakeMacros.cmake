#------------------------------------------------------------------------------
# Copyright 2016 Chuck Seberino
#
# This file is part of CCT.
#
# CCT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CCT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CCT.  If not, see <http://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# cct_add_library(lib)
#
# @param lib   Name of library and attempted subdirectory to use
#
# See __cctGlobForFiles for usage
# If the SHARED keyword is given, it will generate a DLL on windows.
#------------------------------------------------------------------------------
macro (cct_add_library lib)
    __cctGlobForFiles(_srcs ${ARGV})
    add_library(${lib} ${_srcs})

    if (folderDir)
        set_target_properties(${lib} PROPERTIES FOLDER ${folderDir})
    endif ()

    cct_install_debug(${lib})
endmacro ()


#------------------------------------------------------------------------------
# cct_add_executable(exe)
#
# @param exe   Name of executable and subdirectory to use
#
# See __cctGlobForFiles for usage
#------------------------------------------------------------------------------
macro (cct_add_executable exe)
    __cctGlobForFiles(_srcs ${ARGV})
    add_executable(${exe} ${_srcs})
    if (folderDir)
        set_target_properties(${exe} PROPERTIES FOLDER ${folderDir})
    endif ()

    cct_install_debug(${exe})
endmacro ()


#------------------------------------------------------------------------------
# cct_cuda_add_library(lib ...)

# Wrapper for calling cuda_add_library(lib ...) followed by cct_install_debug.
#------------------------------------------------------------------------------
macro (cct_cuda_add_library lib)
    cuda_add_library(${lib} ${ARGN})
    cct_install_debug(${lib})
endmacro ()

#------------------------------------------------------------------------------
# cct_cuda_add_executable(exe ...)

# Wrapper for calling cuda_add_executable(exe ...) followed by cct_install_debug.
#------------------------------------------------------------------------------
macro (cct_cuda_add_executable exe)
    __cctGlobForFiles(_srcs ${ARGV})
    cuda_add_executable(${exe} ${_srcs})
    cct_install_debug(${exe})
endmacro ()

#------------------------------------------------------------------------------
# cct_install_debug(tgt)
#
# @param tgt   Name of target binary
#
# Splits out debug information from target and generates a unique id to link
# with.
#------------------------------------------------------------------------------
macro (cct_install_debug tgt)
    if (CMAKE_HOST_UNIX)
        # Need this on a separate line to keep MSVC generator happy
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
            # Set default name for our debug file
            set(dtgt ${tgt}.debug)
            set_target_properties(${tgt} PROPERTIES LINK_FLAGS "-Wl,--build-id")

            # Create a separate file containing debug info and link to orig.
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND objcopy --only-keep-debug $<TARGET_FILE:${tgt}> ${dtgt}
                COMMAND objcopy --add-gnu-debuglink=${dtgt} $<TARGET_FILE:${tgt}> 
            )
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${dtgt} DESTINATION bin/.debug)
        endif ()
    endif ()
endmacro ()


#------------------------------------------------------------------------------
# cct_add_docs(target doxyString)
#
# @param target     Name of target associated with documentation.  Used at a
#                   Makefile target or Visual Studio project name.
# @param doxyString Optional extra parameters to pass to doxygen.  Appended to
#                   the end of the configuration file.
#
#------------------------------------------------------------------------------
macro (cct_add_docs target doxyString)
    if (NOT DOXYGEN_FOUND)
        return()
    endif ()

    set (doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxy.config)
    set (doxyConfig ${CMAKE_CURRENT_SOURCE_DIR}/doxy.config)

    configure_file(${doxyConfig} ${doxyfile})

    # Add override and customization commands
    file(APPEND ${doxyfile} ${doxyString})
    set (docDir ${CMAKE_BINARY_DIR}/html)
	
    add_custom_command(
    	OUTPUT  ${docDir}/index.html
	COMMAND ${DOXYGEN_EXECUTABLE}
	ARGS    ${doxyfile}
	DEPENDS ${doxyfile}
	COMMENT "Generating html documentation"
    )
    add_custom_target(${target} DEPENDS "${docDir}/index.html")
    set_target_properties(${target} PROPERTIES FOLDER "docs")
endmacro (cct_add_docs doxyString)


#------------------------------------------------------------------------------
# __cctGlobForFiles(dir)
#
# @param dir Name of library and attempted subdirectory to use
#
# Will first try to get files from the same-named subdirectory.  If the subdir
# doesn't exist and there is an additional argument that is a dir,
# then it will use that.  Otherwise it will use the current directory.  It
# collects all .c, .cpp, .cxx, .h, .hpp, and .hxx files.
#
#------------------------------------------------------------------------------
macro (__cctGlobForFiles _srcs dir)
    # Because of some CMake quirk with ARGN/ARGV, we store it in a real list
    # and then remove our fixed args before processing
    set(rest ${ARGV})
    list(REMOVE_AT rest 0 1)
    set(subDir)

    # First check for same named subdir.
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        set(subDir ${dir}/)
    else ()
        # Check if user specified a subdir.
        if (${ARGC} GREATER 2)
            set(optDir ${ARGV2})
            if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${optDir})
                # Remove entry and use.
                list(REMOVE_AT rest 0)
                set(subDir ${optDir}/)
            endif ()
        endif ()
    endif ()

    # Add folders to collate projects by subdirectories
    string(REPLACE "${CMAKE_SOURCE_DIR}" "" folderDir "${CMAKE_CURRENT_SOURCE_DIR}/${subDir}")
    string(SUBSTRING "${folderDir}" 1 -1 folderDir)       # Remove slash prefix
    get_filename_component(folderDir "${folderDir}" PATH) # Remove trailing project name
    set(folderDir "${folderDir}" PARENT_SCOPE)            # Add var to parent scope

    # See if ONLY keyword have been specified
    list(FIND rest ONLY onlyIndex)
    if (NOT ${onlyIndex} EQUAL -1)
        # We don't perform globbing.  Only process what has been specified.
        list(REMOVE_AT rest ${onlyIndex})
        set(srcs)
    else ()
        file(GLOB srcs
            ${subDir}*.c   ${subDir}*.h
            ${subDir}*.cpp ${subDir}*.hpp
            ${subDir}*.cxx ${subDir}.hxx)
        # Prepend additional sources to list, which also handles Windows DLL case
        # where SHARED is passed for library creation.
    endif ()
    set(${_srcs} ${rest} ${srcs})
endmacro ()
