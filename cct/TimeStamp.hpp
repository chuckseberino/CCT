#ifndef CCT_TIMESTAMP_HPP
#define CCT_TIMESTAMP_HPP
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
//
/// @file TimeStamp.hpp
/// @class CCT::TimeStamp
/// @brief System independent wrapper for interval timing statistics.
//-----------------------------------------------------------------------------
#include <chrono>

namespace CCT {

class TimeStamp
{
public:
    typedef std::chrono::steady_clock Clock;
    typedef Clock::time_point TimePoint;
    typedef std::chrono::duration<double> Seconds;

    /// Default constructor
    TimeStamp() : timestamp(Clock::now()) {}

    /// Resets counter to now.
    inline TimeStamp& reset()
    {
        timestamp = Clock::now();
        return *this;
    }

    inline double elapsed() const
    {
        return Seconds(Clock::now() - timestamp).count();
    }

    /// Calculate time difference
    /// @param[in] t TimeStamp to subtract
    /// @return double Difference of two TimeStamps in seconds
    inline double operator-(const TimeStamp& t) const
    {
        return Seconds(timestamp - t.timestamp).count();
    }

protected:
    TimePoint timestamp; //!< Starting time value
};

} // namespace CCT

#endif // CCT_TIMESTAMP_HPP
