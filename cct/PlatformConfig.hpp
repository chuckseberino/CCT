#ifndef CCT_PLATFORMCONFIG_HPP
#define CCT_PLATFORMCONFIG_HPP
//-----------------------------------------------------------------------------
// Copyright 2016 Chuck Seberino
//
// This file is part of CCT.
//
// CCT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CCT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CCT.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------

// Define platform type
#if defined(_MSC_VER)
#   define CCT_MSVC 1
#else
#   if !defined(__GNUC__)
#      warning "Unsupported compiler - defaulting to GNU/Linux"
#   endif
#   define CCT_GCC 1
#endif

// Declaration specification for Windows DLL generation.
#if defined(CCT_MSVC)
#   define CCT_EXPORT __declspec(dllexport)
#   define CCT_IMPORT __declspec(dllimport)
#else
#   define CCT_EXPORT
#   define CCT_IMPORT
#endif

#endif // CCT_PLATFORMCONFIG_HPP
