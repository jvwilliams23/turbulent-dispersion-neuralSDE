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
    Foam::DispersionModelHW

Description

\*---------------------------------------------------------------------------*/

#ifndef DispersionModelHW_H
#define DispersionModelHW_H

#include "IOdictionary.H"
#include "autoPtr.H"
#include "runTimeSelectionTables.H"
#include "CloudSubModelBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class DispersionModelHW Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class DispersionModelHW
:
    public CloudSubModelBase<CloudType>
{
public:

    //- Runtime type information
    TypeName("dispersionModelHW");


    // Declare runtime constructor selection table

    declareRunTimeSelectionTable
    (
        autoPtr,
        DispersionModelHW,
        dictionary,
        (
            const dictionary& dict,
            CloudType& owner
        ),
        (dict, owner)
    );


    // Constructors

        //- Construct null from owner
        DispersionModelHW(CloudType& owner);

        //- Construct from components
        DispersionModelHW
        (
            const dictionary& dict,
            CloudType& owner,
            const word& type
        );

        //- Construct copy
        DispersionModelHW(const DispersionModelHW<CloudType>& dm);

        //- Construct and return a clone
        virtual autoPtr<DispersionModelHW<CloudType>> clone() const = 0;


    //- Destructor
    virtual ~DispersionModelHW();


    //- Selector
    static autoPtr<DispersionModelHW<CloudType>> New
    (
        const dictionary& dict,
        CloudType& owner
    );


    // Member Functions

        //- Update (disperse particles)
        virtual vector update
        (
            const TrackCloudType& cloud
            const scalar dt,
            const label celli,
            const vector& U,
            const vector& Uc,
            vector& UTurb,
            scalar& tTurb
        ) = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define makeDispersionModelHW(CloudType)                                         \
                                                                               \
    typedef Foam::CloudType::kinematicHWCloudType kinematicHWCloudType;            \
    defineTemplateTypeNameAndDebug                                             \
        (Foam::DispersionModelHW<kinematicHWCloudType>, 0);                        \
                                                                               \
    namespace Foam                                                             \
    {                                                                          \
        defineTemplateRunTimeSelectionTable                                    \
        (                                                                      \
            DispersionModelHW<kinematicHWCloudType>,                               \
            dictionary                                                         \
        );                                                                     \
    }

#define makeDispersionModelHWType(SS, CloudType)                                 \
                                                                               \
    typedef Foam::CloudType::kinematicHWCloudType kinematicHWCloudType;            \
    defineNamedTemplateTypeNameAndDebug(Foam::SS<kinematicHWCloudType>, 0);      \
                                                                               \
    Foam::DispersionModelHW<kinematicHWCloudType>::                                \
        adddictionaryConstructorToTable<Foam::SS<kinematicHWCloudType>>          \
            add##SS##CloudType##kinematicHWCloudType##ConstructorToTable_;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "DispersionModelHW.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
