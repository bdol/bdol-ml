"""
The greedy algorithm for submodular maximization.

==============
Copyright Info
==============
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright Brian Dolhansky 2014
bdolmail@gmail.com
"""

def greedy_submodular_max(X, V, k, F, args):
    A = set()
    for i in range(0, k):
        V_rem = V-A

        max_f = 0.0
        max_j = -1
        for j in V_rem:
            if j%1000==0:
                print j
            A_next = set(A)
            A_next.add(j)

            f = F(X, A_next, args)
            if f > max_f:
                max_f = f
                max_j = j

        A.add(max_j)
        print "Added {0} with max_f {1}".format(max_j, max_f)

    return A
