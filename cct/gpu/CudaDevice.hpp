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
#pragma once

#include <map>

namespace CCT {

class Device
{
public:
    /// Constructor.
    /// @param[in] deviceID Specify device to use, or choose default
    explicit Device(int deviceID);

    /// Destructor. Releases all CUDA-created memory properly.
    ~Device();

    /// Return the number of GPU devices available.
    /// @return int Number of CUDA capable devices available
    static int numDevices();

    /// Sets the CUDA device for this thread
    void setDevice();

    /// Retrieve the device ID in use. A value of -1 denotes no valid GPU.
    /// @return int GPU Device ID in use
    int ID() const { return deviceID; }

    /// Retrieve the clock rate of this device.
    /// @return int Device clock rate in kHz
    int clockRate() const { return clockRateInKHz; }

private:
    /// Keep track of device objects available on this system
    typedef std::map<int, int> DeviceMap;

    int deviceID;               //!< Device in use for this object
    int clockRateInKHz;         //!< Device clock rate in kHz
    static DeviceMap deviceMap; //!< Complete list of valid devices
};

} // namespace CCT
