/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
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
    Foam::CollidingHWCloud

Description
    Adds coolisions to kinematic clouds

SourceFiles
    CollidingHWCloudI.H
    CollidingHWCloud.C

\*---------------------------------------------------------------------------*/

#ifndef CollidingHWCloud_H
#define CollidingHWCloud_H

#include "particle.H"
#include "Cloud.H"
#include "IOdictionary.H"
#include "autoPtr.H"
#include "fvMesh.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of classes

template<class CloudType>
class CollisionModel;

/*---------------------------------------------------------------------------*\
                       Class CollidingHWCloud Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class CollidingHWCloud
:
    public CloudType
{
public:

    // Public typedefs

        //- Type of cloud this cloud was instantiated for
        typedef CloudType cloudType;

        //- Type of parcel the cloud was instantiated for
        typedef typename CloudType::particleType parcelType;

        //- Convenience typedef for this cloud type
        typedef CollidingHWCloud<CloudType> collidingCloudType;


private:

    // Private data

        //- Cloud copy pointer
        autoPtr<CollidingHWCloud<CloudType>> cloudCopyPtr_;


    // Private Member Functions

        //- Disallow default bitwise copy construct
        CollidingHWCloud(const CollidingHWCloud&);

        //- Disallow default bitwise assignment
        void operator=(const CollidingHWCloud&);


protected:

    // Protected data

        //- Thermo parcel constant properties
        typename parcelType::constantProperties constProps_;


        // References to the cloud sub-models

            //- Collision model
            autoPtr<CollisionModel<CollidingHWCloud<CloudType>>>
                collisionModel_;


        // Initialisation

            //- Set cloud sub-models
            void setModels();


        // Cloud evolution functions

            //- Move-collide particles
            template<class TrackCloudType>
            void moveCollide
            (
                TrackCloudType& cloud,
                typename parcelType::trackingData& td,
                const scalar deltaT
            );

            //- Reset state of cloud
            void cloudReset(CollidingHWCloud<CloudType>& c);


public:

    // Constructors

        //- Construct given carrier gas fields
        CollidingHWCloud
        (
            const word& cloudName,
            const volScalarField& rho,
            const volVectorField& U,
            const volScalarField& mu,
            const dimensionedVector& g,
            bool readFields = true
        );

        //- Copy constructor with new name
        CollidingHWCloud
        (
            CollidingHWCloud<CloudType>& c,
            const word& name
        );

        //- Copy constructor with new name - creates bare cloud
        CollidingHWCloud
        (
            const fvMesh& mesh,
            const word& name,
            const CollidingHWCloud<CloudType>& c
        );

        //- Construct and return clone based on (this) with new name
        virtual autoPtr<Cloud<parcelType>> clone(const word& name)
        {
            return autoPtr<Cloud<parcelType>>
            (
                new CollidingHWCloud(*this, name)
            );
        }

        //- Construct and return bare clone based on (this) with new name
        virtual autoPtr<Cloud<parcelType>> cloneBare(const word& name) const
        {
            return autoPtr<Cloud<parcelType>>
            (
                new CollidingHWCloud(this->mesh(), name, *this)
            );
        }


    //- Destructor
    virtual ~CollidingHWCloud();


    // Member Functions

        // Access

            //- Return a reference to the cloud copy
            inline const CollidingHWCloud& cloudCopy() const;

            //- Return the constant properties
            inline const typename parcelType::constantProperties&
                constProps() const;


            // Sub-models

                //- Return const access to the collision model
                inline const CollisionModel<CollidingHWCloud<CloudType>>&
                    collision() const;

                //- Return reference to the collision model
                inline CollisionModel<CollidingHWCloud<CloudType>>&
                    collision();

        // Check

            //- Total rotational kinetic energy in the system
            inline scalar rotationalKineticEnergyOfSystem() const;


        // Cloud evolution functions

            //- Store the current cloud state
            void storeState();

            //- Reset the current cloud to the previously stored state
            void restoreState();

            //- Evolve the cloud
            void evolve();

            //- Particle motion
            template<class TrackCloudType>
            void motion
            (
                TrackCloudType& cloud,
                typename parcelType::trackingData& td
            );


        // I-O

            //- Print cloud information
            void info();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "CollidingHWCloudI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "CollidingHWCloud.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
