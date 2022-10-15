/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2015 OpenFOAM Foundation
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
    Foam::MPPICHWParcel

Description
    Wrapper around kinematic parcel types to add MPPIC modelling

SourceFiles
    MPPICHWParcelI.H
    MPPICHWParcelTrackingDataI.H
    MPPICHWParcel.C
    MPPICHWParcelIO.C

\*---------------------------------------------------------------------------*/

#ifndef MPPICHWParcel_H
#define MPPICHWParcel_H

#include "particle.H"
#include "labelFieldIOField.H"
#include "vectorFieldIOField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of clases

template<class ParcelType>
class MPPICHWParcel;

template<class Type>
class AveragingMethod;

// Forward declaration of friend functions

template<class ParcelType>
Ostream& operator<<
(
    Ostream&,
    const MPPICHWParcel<ParcelType>&
);

/*---------------------------------------------------------------------------*\
                       Class MPPICHWParcel Declaration
\*---------------------------------------------------------------------------*/

template<class ParcelType>
class MPPICHWParcel
:
    public ParcelType
{
    // Private data

        //- Size in bytes of the fields
        static const std::size_t sizeofFields_;


public:

    class trackingData
    :
        public ParcelType::trackingData
    {

    public:

        enum trackPart
        {
            tpLinearTrack,
            tpDampingNoTrack,
            tpPackingNoTrack,
            tpCorrectTrack,
        };


    public:

        // Private data

            // MPPIC Averages

                //- Volume average
                autoPtr<AveragingMethod<scalar> > volumeAverage_;

                //- Radius average [ volume^(1/3) ]
                autoPtr<AveragingMethod<scalar> > radiusAverage_;

                //- Density average
                autoPtr<AveragingMethod<scalar> > rhoAverage_;

                //- Velocity average
                autoPtr<AveragingMethod<vector> > uAverage_;

                //- Magnitude velocity sqyuared average
                autoPtr<AveragingMethod<scalar> > uSqrAverage_;

                //- Frequency average
                autoPtr<AveragingMethod<scalar> > frequencyAverage_;

                //- Mass average
                autoPtr<AveragingMethod<scalar> > massAverage_;

		/*
		//- Charge average
                autoPtr<AveragingMethod<scalar> > qAverage_;
		
		autoPtr<AveragingMethod<scalar> > rhoeAverage_;
		
		//electric field interpolation
		autoPtr<interpolation<vector> > EInterp_;
		
		//mean charge interpolation
		autoPtr<interpolation<scalar> > qInterp_;
		
		//- Work Function average
                autoPtr<AveragingMethod<scalar> > workFunctionAverage_;
		
		//Work Function interpolation
		autoPtr<interpolation<scalar> > workFunctionInterp_;

		//- Solid velocity interpolation
                autoPtr<interpolation<vector> > usInterp_;
		
		//- Solid velocity interpolation
                autoPtr<interpolation<scalar> > volumeAverageInterp_;
		
		//- Solid velocity interpolation
                autoPtr<interpolation<scalar> > granularTemperatureInterp_;
		
		//- Charge Deficit
                autoPtr<AveragingMethod<scalar> > chargeDefAverage_;		
		autoPtr<interpolation<scalar> > chargeDefInterp_;
		*/
							
            //- Label specifying the current part of the tracking process
            trackPart part_;


    //public:

        //- Constructors

            //- Construct from components
            template<class TrackCloudType>
            inline trackingData
            (
                const TrackCloudType& cloud,
                trackPart part = tpLinearTrack
            );

	    /*
	    inline const interpolation<vector>& EInterp() const;

            //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<scalar>& qInterp() const;
	    
	                //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<scalar>& workFunctionInterp() const;
	    
            //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<vector>& usInterp() const;

            //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<scalar>& volumeAverageInterp() const;

            //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<scalar>& granularTemperatureInterp() const;	

            //- Return conat access to the interpolator for continuous
            //  phase dynamic viscosity field
            inline const interpolation<scalar>& chargeDefInterp() const;
	    */
	
        //- Update the MPPIC averages
		template<class TrackCloudType>
        inline void updateAverages(const TrackCloudType& cloud);
	
        //- Update the MPPIC averages
        template<class TrackCloudType>
        inline void updateChargeDeficit(const TrackCloudType&  cloud);


        //- Access

            //- Const access to the tracking part label
            inline trackPart part() const;

            //- Non const access to the tracking part label
            inline trackPart& part();
    };


protected:

    // Protected data

        //- Velocity correction due to collisions [m/s]
        vector UCorrect_;
	
	/*
	//parcel properties
	scalar charge_;
	
	scalar chargeFlux_;
	
	scalar workFunction_;
	
	vector Ef_;
	
	//parcel location properties
	scalar chargeI_;
	
	scalar workFunctionI_;
	
	scalar chargeDef_;
	
	vector UI_;
	
	scalar granularTemperature_;
	
	scalar volumeFraction_;
	*/
	
public:

    // Static data members
	
        //- Runtime type information
        TypeName("MPPICHWParcel");

        //- String representation of properties
        AddToPropertyList
        (
            ParcelType,
            "(UCorrectx UCorrecty UCorrectz)"
	//    + "charge"
	//    + "workFunction"
        );

	
	bool setValuesCalled;
	
    // Constructors

        //- Construct from owner, position, and cloud owner
        //  Other properties initialised as null
        inline MPPICHWParcel
        (
            const polyMesh& mesh,
            const vector& position,
            const label cellI,
            const label tetFaceI,
            const label tetPtI
        );

        //- Construct from a position and a cell, searching for the rest of the
        //  required topology. Other properties are initialised as null.
        inline MPPICHWParcel
        (
            const polyMesh& mesh,
            const vector& position,
            const label celli
        );

        //- Construct from components
        inline MPPICHWParcel
        (
            const polyMesh& mesh,
            const vector& position,
            const label cellI,
            const label tetFaceI,
            const label tetPtI,
            const label typeId,
            const scalar nParticle0,
            const scalar d0,
            const scalar dTarget0,
            const vector& U0,
            const vector& UCorrect0,
            const typename ParcelType::constantProperties& constProps
        );
	
	/*
	        //- Construct from components
        inline MPPICHWParcel
        (
            const polyMesh& mesh,
            const vector& position,
            const label cellI,
            const label tetFaceI,
            const label tetPtI,
            const label typeId,
            const scalar nParticle0,
            const scalar d0,
            const scalar dTarget0,
            const vector& U0,
            const vector& UCorrect0,
            const typename ParcelType::constantProperties& constProps,
	    const scalar&,
	    const scalar&
        );
	*/
	
        //- Construct from Istream
        MPPICHWParcel
        (
            const polyMesh& mesh,
            Istream& is,
            bool readFields = true
        );

        //- Construct as a copy
        MPPICHWParcel(const MPPICHWParcel& p);

        //- Construct as a copy
        MPPICHWParcel(const MPPICHWParcel& p, const polyMesh& mesh);

        //- Construct and return a (basic particle) clone
        virtual autoPtr<particle> clone() const
        {
            return autoPtr<particle>(new MPPICHWParcel(*this));
        }

        //- Construct and return a (basic particle) clone
        virtual autoPtr<particle> clone(const polyMesh& mesh) const
        {
            return autoPtr<particle>(new MPPICHWParcel(*this, mesh));
        }
        
	/*
	template<class TrackCloudType>
	void updateCharge( TrackCloudType& cloud,
						trackingData& td, const scalar trackTime );
	*/
	
        //- Factory class to read-construct particles used for
        //  parallel transfer
        class iNew
        {
            const polyMesh& mesh_;

        public:

            iNew(const polyMesh& mesh)
            :
                mesh_(mesh)
            {}

            autoPtr<MPPICHWParcel<ParcelType> > operator()(Istream& is) const
            {
                return autoPtr<MPPICHWParcel<ParcelType> >
                (
                    new MPPICHWParcel<ParcelType>(mesh_, is, true)
                );
            }
        };


    // Member Functions
        
	    /*
	    //- Set cell values
            template<class TrackData>
            void setCellValues
            (
                TrackData& td,
                const scalar dt,
                const label cellI
            );
	    */
	
        // Access

            //- Return const access to correction velocity
            inline const vector& UCorrect() const;

            //- Return access to correction velocity
            inline vector& UCorrect();
	    
	    /*
	    //- Return const access to correction velocity
            inline const scalar& charge() const;

            //- Return access to correction velocity
            inline scalar& charge();

	    //- Return const access to correction velocity
            inline const scalar& chargeFlux() const;

            //- Return access to correction velocity
            inline scalar& chargeFlux();

	    //- Return const access to correction velocity
            inline const scalar& chargeI() const;

            //- Return access to correction velocity
            inline scalar& chargeI();

	    //- Return const access to correction velocity
            inline const scalar& workFunctionI() const;

            //- Return access to correction velocity
            inline scalar& workFunctionI();

	    //- Return const access to correction velocity
            inline const scalar& workFunction() const;

            //- Return access to correction velocity
            inline scalar& workFunction();

	    //- Return const access to correction velocity
            inline const scalar& chargeDef() const;

            //- Return access to correction velocity
            inline scalar& chargeDef();

	    //- Return const access to correction velocity
            inline const vector& UI() const;

            //- Return access to correction velocity
            inline vector& UI();
	    
	    //- Return access to correction velocity
            inline scalar& granularTemperature();

	    //- Return const access to correction velocity
            inline const scalar& granularTemperature() const;
	    
	    //- Return access to correction velocity
            inline scalar& volumeFraction();

	    //- Return const access to correction velocity
            inline const scalar& volumeFraction() const;
	    
	    
	    //- Return const access to correction velocity
            inline const vector& Ef() const;

            //- Return access to correction velocity
            inline vector& Ef();
	    */
		
        // Tracking

            //- Move the parcel
			template<class TrackCloudType>
            bool move
            (
                TrackCloudType& cloud,
                trackingData& td,
                const scalar trackTime
            );


    // Friend Functions

        // I-O

            //- Read
            template<class CloudType>
            static void readFields(CloudType& c);

            //- Write
            template<class CloudType>
            static void writeFields(const CloudType& c);


        // Ostream operator

            friend Ostream& operator<< <ParcelType>
            (
                Ostream&,
                const MPPICHWParcel<ParcelType>&
            );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "MPPICHWParcelI.H"
#include "MPPICHWParcelTrackingDataI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "MPPICHWParcel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
