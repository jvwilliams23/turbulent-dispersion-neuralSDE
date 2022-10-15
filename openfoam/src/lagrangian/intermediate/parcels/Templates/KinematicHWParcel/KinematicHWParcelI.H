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

#include "mathematicalConstants.H"

using namespace Foam::constant::mathematical;

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
inline
Foam::KinematicHWParcel<ParcelType>::constantProperties::constantProperties()
:
    dict_(dictionary::null),
    parcelTypeId_(dict_, -1),
    rhoMin_(dict_, 0.0),
    rho0_(dict_, 0.0),
    minParcelMass_(dict_, 0.0)
{}


template<class ParcelType>
inline Foam::KinematicHWParcel<ParcelType>::constantProperties::constantProperties
(
    const constantProperties& cp
)
:
    dict_(cp.dict_),
    parcelTypeId_(cp.parcelTypeId_),
    rhoMin_(cp.rhoMin_),
    rho0_(cp.rho0_),
    minParcelMass_(cp.minParcelMass_)
{}


template<class ParcelType>
inline Foam::KinematicHWParcel<ParcelType>::constantProperties::constantProperties
(
    const dictionary& parentDict
)
:
    dict_(parentDict.subOrEmptyDict("constantProperties")),
    parcelTypeId_(dict_, "parcelTypeId", -1),
    rhoMin_(dict_, "rhoMin", 1e-15),
    rho0_(dict_, "rho0"),
    minParcelMass_(dict_, "minParcelMass", 1e-15)
{}


template<class ParcelType>
inline Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const polyMesh& owner,
    const barycentric& coordinates,
    const label celli,
    const label tetFacei,
    const label tetPti
)
:
    ParcelType(owner, coordinates, celli, tetFacei, tetPti),
    active_(true),
    typeId_(-1),
    nParticle_(0),
    d_(0.0),
    dTarget_(0.0),
    U_(Zero),
    rho_(0.0),
    age_(0.0),
    tTurb_(0.0),
    UTurb_(Zero),
    UTurbN_(Zero), // JW 05/02/21
    UTurbPred_(Zero),
    UN_(Zero), // JW 05/02/21
    UPred_(Zero),
    UNext_(Zero),
    s_(Zero),
    G1_(Zero),
    G2_(Zero),
    stochastic_(false),
    rebound_(false)
{}


template<class ParcelType>
inline Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const polyMesh& owner,
    const vector& position,
    const label celli
)
:
    ParcelType(owner, position, celli),
    active_(true),
    typeId_(-1),
    nParticle_(0),
    d_(0.0),
    dTarget_(0.0),
    U_(Zero),
    rho_(0.0),
    age_(0.0),
    tTurb_(0.0),
    UTurb_(Zero),
    UTurbN_(Zero), // JW 05/02/21
    UTurbPred_(Zero),
    UN_(Zero), //JW 05/02/21
    UPred_(Zero),
    UNext_(Zero),
    s_(Zero),
    G1_(Zero),
    G2_(Zero),
    stochastic_(false),
    rebound_(false)

{}


template<class ParcelType>
inline Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const polyMesh& owner,
    const barycentric& coordinates,
    const label celli,
    const label tetFacei,
    const label tetPti,
    const label typeId,
    const scalar nParticle0,
    const scalar d0,
    const scalar dTarget0,
    const vector& U0,
    const constantProperties& constProps
)
:
    ParcelType(owner, coordinates, celli, tetFacei, tetPti),
    active_(true),
    typeId_(typeId),
    nParticle_(nParticle0),
    d_(d0),
    dTarget_(dTarget0),
    U_(U0),
    rho_(constProps.rho0()),
    age_(0.0),
    tTurb_(0.0),
    UTurb_(Zero),
    UTurbN_(Zero), // JW 05/02/21
    UTurbPred_(Zero),
    UN_(Zero), // JW 05/02/21
    UPred_(Zero),
    UNext_(Zero),
    s_(Zero),
    G1_(Zero),
    G2_(Zero),
    stochastic_(false),
    rebound_(false)
{}


// * * * * * * * * * constantProperties Member Functions * * * * * * * * * * //

template<class ParcelType>
inline const Foam::dictionary&
Foam::KinematicHWParcel<ParcelType>::constantProperties::dict() const
{
    return dict_;
}


template<class ParcelType>
inline Foam::label
Foam::KinematicHWParcel<ParcelType>::constantProperties::parcelTypeId() const
{
    return parcelTypeId_.value();
}


template<class ParcelType>
inline Foam::scalar
Foam::KinematicHWParcel<ParcelType>::constantProperties::rhoMin() const
{
    return rhoMin_.value();
}


template<class ParcelType>
inline Foam::scalar
Foam::KinematicHWParcel<ParcelType>::constantProperties::rho0() const
{
    return rho0_.value();
}


template<class ParcelType>
inline Foam::scalar
Foam::KinematicHWParcel<ParcelType>::constantProperties::minParcelMass() const
{
    return minParcelMass_.value();
}


// * * * * * * * KinematicHWParcel Member Functions  * * * * * * * //

template<class ParcelType>
inline bool Foam::KinematicHWParcel<ParcelType>::active() const
{
    return active_;
}


template<class ParcelType>
inline Foam::label Foam::KinematicHWParcel<ParcelType>::typeId() const
{
    return typeId_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::nParticle() const
{
    return nParticle_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::d() const
{
    return d_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::dTarget() const
{
    return dTarget_;
}


template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::U() const
{
    return U_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::rho() const
{
    return rho_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::age() const
{
    return age_;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::tTurb() const
{
    return tTurb_;
}


template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurb() const
{
    return UTurb_;
}


// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurbN() const
{
    return UTurbN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurbPred() const
{
    return UTurbPred_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UN() const
{
    return UN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UPred() const
{
    return UPred_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UNext() const
{
    return UNext_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UcTildeN() const
{
    return UcTildeN_;
}


// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UpTildeN() const
{
    return UpTildeN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UsTildeN() const
{
    return UsTildeN_;
}


// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::PGradTildeN() const
{
    return PGradTildeN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::scalar& Foam::KinematicHWParcel<ParcelType>::kN() const
{
    return kN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::scalar& Foam::KinematicHWParcel<ParcelType>::epsilonN() const
{
    return epsilonN_;
}


// JW 05/02/21
template<class ParcelType>
inline const Foam::tensor& Foam::KinematicHWParcel<ParcelType>::UcGradN() const
{
    return UcGradN_;
}


// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::UcLaplacianN() const
{
    return UcLaplacianN_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::s() const
{
    return s_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::scalar& Foam::KinematicHWParcel<ParcelType>::kHatOverk() const
{
    return kHatOverk_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::G1() const
{
    return G1_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::G2() const
{
    return G2_;
}

template<class ParcelType>
inline bool Foam::KinematicHWParcel<ParcelType>::stochastic() const
{
    return stochastic_;
}

template<class ParcelType>
inline bool Foam::KinematicHWParcel<ParcelType>::rebound() const
{
    return rebound_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::vector& Foam::KinematicHWParcel<ParcelType>::reboundnw() const
{
    return reboundnw_;
}


/*
// JW 05/02/21
template<class ParcelType>
inline Foam::barycentric Foam::KinematicHWParcel<ParcelType>::initCoordinates() const
{
    return coordinatesHW_;
}

// JW 05/02/21
template<class ParcelType>
inline const Foam::tetIndices& Foam::KinematicHWParcel<ParcelType>::initTetIndices() const
{
    return tetIndices(celliHW_, tetFaceiHW_, tetPtiHW_);
}*/

template<class ParcelType>
inline bool& Foam::KinematicHWParcel<ParcelType>::active()
{
    return active_;
}


template<class ParcelType>
inline Foam::label& Foam::KinematicHWParcel<ParcelType>::typeId()
{
    return typeId_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::nParticle()
{
    return nParticle_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::d()
{
    return d_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::dTarget()
{
    return dTarget_;
}


template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::U()
{
    return U_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::rho()
{
    return rho_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::age()
{
    return age_;
}


template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::tTurb()
{
    return tTurb_;
}


template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurb()
{
    return UTurb_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurbN()
{
    return UTurbN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UTurbPred()
{
    return UTurbPred_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UN()
{
    return UN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UPred()
{
    return UPred_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UNext()
{
    return UNext_;
}

template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UcTildeN()
{
    return UcTildeN_;
}


// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UpTildeN()
{
    return UpTildeN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UsTildeN()
{
    return UsTildeN_;
}


// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::PGradTildeN()
{
    return PGradTildeN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::kN()
{
    return kN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::epsilonN()
{
    return epsilonN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::scalar& Foam::KinematicHWParcel<ParcelType>::kHatOverk()
{
    return kHatOverk_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::tensor& Foam::KinematicHWParcel<ParcelType>::UcGradN()
{
    return UcGradN_;
}


// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::UcLaplacianN() 
{
    return UcLaplacianN_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::s()
{
    return s_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::G1()
{
    return G1_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::G2()
{
    return G2_;
}

template<class ParcelType>
inline bool& Foam::KinematicHWParcel<ParcelType>::stochastic()
{   
    return stochastic_;
}

template<class ParcelType>
inline bool& Foam::KinematicHWParcel<ParcelType>::rebound()
{   
    return rebound_;
}

// JW 05/02/21
template<class ParcelType>
inline Foam::vector& Foam::KinematicHWParcel<ParcelType>::reboundnw()
{
    return reboundnw_;
}


// JW 05/02/21
/*template<class ParcelType>
inline Foam::tetIndices& Foam::KinematicHWParcel<ParcelType>::initTetIndices()
{
    return tetIndices(celliHW_, tetFaceiHW_, tetPtiHW_);
}*/


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::massCell
(
    const trackingData& td
) const
{
    return td.rhoc()*this->mesh().cellVolumes()[this->cell()];
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::mass() const
{
    return rho_*volume();
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::momentOfInertia() const
{
    return 0.1*mass()*sqr(d_);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::volume() const
{
    return volume(d_);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::volume(const scalar d)
{
    return pi/6.0*pow3(d);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::areaP() const
{
    return areaP(d_);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::areaP(const scalar d)
{
    return 0.25*areaS(d);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::areaS() const
{
    return areaS(d_);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::areaS(const scalar d)
{
    return pi*d*d;
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::Re
(
    const trackingData& td
) const
{
    return Re(td.rhoc(), U_, td.Uc(), d_, td.muc());
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::Re
(
    const scalar rhoc,
    const vector& U,
    const vector& Uc,
    const scalar d,
    const scalar muc
)
{
    return rhoc*mag(U - Uc)*d/max(muc, rootVSmall);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::We
(
    const trackingData& td,
    const scalar sigma
) const
{
    return We(td.rhoc(), U_, td.Uc(), d_, sigma);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::We
(
    const scalar rhoc,
    const vector& U,
    const vector& Uc,
    const scalar d,
    const scalar sigma
)
{
    return rhoc*magSqr(U - Uc)*d/max(sigma, rootVSmall);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::Eo
(
    const trackingData& td,
    const scalar sigma
) const
{
    return Eo(td.g(), rho_, td.rhoc(), U_, d_, sigma);
}


template<class ParcelType>
inline Foam::scalar Foam::KinematicHWParcel<ParcelType>::Eo
(
    const vector& g,
    const scalar rho,
    const scalar rhoc,
    const vector& U,
    const scalar d,
    const scalar sigma
)
{
    const vector dir = U/max(mag(U), rootVSmall);
    return mag(g & dir)*(rho - rhoc)*sqr(d)/max(sigma, rootVSmall);
}


// ************************************************************************* //
