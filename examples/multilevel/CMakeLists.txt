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

cct_cuda_add_executable(multilevelTest multilevelTest.cu)
target_link_libraries(multilevelTest gpu)
