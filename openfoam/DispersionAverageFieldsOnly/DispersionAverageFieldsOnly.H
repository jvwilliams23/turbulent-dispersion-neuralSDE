/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::DispersionAverageFieldsOnly

Description
    The velocity is perturbed in random direction, with a
    Gaussian random number distribution with variance sigma.
    where sigma is defined below

\*---------------------------------------------------------------------------*/

#ifndef DispersionAverageFieldsOnly_H
#define DispersionAverageFieldsOnly_H

#include "DispersionRASModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                   Class DispersionAverageFieldsOnly Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class DispersionAverageFieldsOnly
:
    public DispersionRASModel<CloudType>
{
    //-Eulerian particle velocity field
    volVectorField UpTilde_;
    //- Name of carrier velocity field
    const word UName_;
    const word initCond_;
    const bool fillEmptyCells_;

public:

    //- Runtime type information
    TypeName("dispersionAverageFieldsOnly");
    


    // Constructors

        //- Construct from components
        DispersionAverageFieldsOnly(const dictionary& dict, CloudType& owner);

        //- Construct copy
        DispersionAverageFieldsOnly(const DispersionAverageFieldsOnly<CloudType>& dm);

        //- Construct and return a clone
        virtual autoPtr<DispersionModel<CloudType>> clone() const
        {
            return autoPtr<DispersionModel<CloudType>>
            (
                new DispersionAverageFieldsOnly<CloudType>(*this)
            );
        }


    //- Destructor
    virtual ~DispersionAverageFieldsOnly();

        //- Cache carrier fields
        virtual void cacheFields(const bool store);

        //- Update (disperse particles)
        virtual vector update
        (
            const scalar dt,
            const label celli,
            const vector& U,
            const vector& Uc,
            vector& UTurb,
            scalar& tTurb
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#ifdef NoRepository
    #include "DispersionAverageFieldsOnly.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
