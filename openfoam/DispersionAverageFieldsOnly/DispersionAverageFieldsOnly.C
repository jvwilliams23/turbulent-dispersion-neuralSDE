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

\*---------------------------------------------------------------------------*/

#include "DispersionAverageFieldsOnly.H"
#include "constants.H"

using namespace Foam::constant::mathematical;

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::DispersionAverageFieldsOnly<CloudType>::DispersionAverageFieldsOnly
(
    const dictionary& dict,
    CloudType& owner
)
:
    DispersionRASModel<CloudType>(dict, owner),
    UpTilde_
    (
        IOobject
        (
            "UpTilde",
            this->owner().mesh().time().timeName(),
            this->owner().mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->owner().mesh(),
        dimensionedVector("0", dimLength/dimTime, Zero)
    ),
    UName_(dict.subDict("DispersionAverageFieldsOnlyCoeffs").lookupOrDefault<word>("U", "U.air")),
    initCond_(dict.subDict("DispersionAverageFieldsOnlyCoeffs").lookupOrDefault<word>("init","0")),
    fillEmptyCells_(dict.subDict("DispersionAverageFieldsOnlyCoeffs").lookupOrDefault("fillEmptyCells",false))
{}


template<class CloudType>
Foam::DispersionAverageFieldsOnly<CloudType>::DispersionAverageFieldsOnly
(
    const DispersionAverageFieldsOnly<CloudType>& dm
)
:
    DispersionRASModel<CloudType>(dm),
    UpTilde_(dm.UpTilde_),
    UName_(dm.UName_),
    initCond_(dm.initCond_),
    fillEmptyCells_(dm.fillEmptyCells_)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class CloudType>
Foam::DispersionAverageFieldsOnly<CloudType>::~DispersionAverageFieldsOnly()
{
    cacheFields(false);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::DispersionAverageFieldsOnly<CloudType>::cacheFields(const bool store)
{
    const objectRegistry& mesh = this->owner().mesh();

    const volScalarField& mu = mesh.template
        lookupObject<volScalarField>("mu.air");
    const volScalarField& rho = mesh.template
        lookupObject<volScalarField>("rho.air");

    const volVectorField& Uc = mesh.template
        lookupObject<volVectorField>(UName_);
    const scalar deltaT = this->owner().mesh().time().deltaT().value();

    // Get Eulerian fields of mass averaged Lagrangian properties
    if (store)
    {
#       include "AverageLagrangian.H"
        UpTilde_.primitiveFieldRef() = upTildeAverage.primitiveField();

        // if chosen to fill in cells with no particles for averaging
        // maybe will improve rms statistics
        if (fillEmptyCells_)
        {
            // Info << "Setting UpTilde in cells with no particles as Uc (try to reduce noise)" << endl;
            const polyMesh& pmesh = this->owner().mesh();
            List<DynamicList<typename CloudType::parcelType*>>& cellOccupancy = 
                this->owner().cellOccupancy();

            //-map Field data to volField 
            forAll(pmesh.cells(), celli)
            {
                // when cell has no particles for averaging, 
                // set Eulerian particle fields to Uc
                if ( cellOccupancy[celli].size() == 0 )
                {
                    UpTilde_[celli] = Uc[celli];
                }
            }
        }
    }
}

template<class CloudType>
Foam::vector Foam::DispersionAverageFieldsOnly<CloudType>::update
(
    const scalar dt,
    const label celli,
    const vector& U,
    const vector& Uc,
    vector& UTurb,
    scalar& tTurb
)
{
    return Uc;
}


// ************************************************************************* //
