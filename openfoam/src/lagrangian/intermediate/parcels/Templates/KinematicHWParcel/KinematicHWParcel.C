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

#include "KinematicHWParcel.H"
#include "forceSuSp.H"
#include "integrationScheme.H"
#include "meshTools.H"

//#include "UfSeenByParticlesLES.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::label Foam::KinematicHWParcel<ParcelType>::maxTrackAttempts = 1;


// * * * * * * * * * * *  Protected Member Functions * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::setCellValues
(
    TrackCloudType& cloud,
    trackingData& td
)
{
    tetIndices tetIs = this->currentTetIndices();

    td.rhoc() = td.rhoInterp().interpolate(this->coordinates(), tetIs);

    if (td.rhoc() < cloud.constProps().rhoMin())
    {
        if (debug)
        {
            WarningInFunction
                << "Limiting observed density in cell " << this->cell()
                << " to " << cloud.constProps().rhoMin() <<  nl << endl;
        }

        td.rhoc() = cloud.constProps().rhoMin();
    }

    td.Uc() = td.UInterp().interpolate(this->coordinates(), tetIs);

    td.muc() = td.muInterp().interpolate(this->coordinates(), tetIs);
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::calcDispersion
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    td.Uc() = cloud.dispersion().update
    (
    //    cloud,
        dt,
        this->cell(),
        U_,
        td.Uc(),
        UTurb_,
        tTurb_
    );
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::cellValueSourceCorrection
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    td.Uc() += cloud.UTrans()[this->cell()]/massCell(td);
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::calc
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    // Define local properties at beginning of time step
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    const scalar np0 = nParticle_;
    const scalar mass0 = mass();

    // Reynolds number
    const scalar Re = this->Re(td);


    // Sources
    //~~~~~~~~

    // Explicit momentum source for particle
    vector Su = Zero;

    // Linearised momentum source coefficient
    scalar Spu = 0.0;

    // Momentum transfer from the particle to the carrier phase
    vector dUTrans = Zero;


    // Motion
    // ~~~~~~

    // Calculate new particle velocity
    this->U_ =
        calcVelocity(cloud, td, dt, Re, td.muc(), mass0, Su, dUTrans, Spu);


    // Accumulate carrier phase source terms
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (cloud.solution().coupled())
    {
        // Update momentum transfer
        cloud.UTrans()[this->cell()] += np0*dUTrans;

        // Update momentum transfer coefficient
        cloud.UCoeff()[this->cell()] += np0*Spu;
    }
}


template<class ParcelType>
template<class TrackCloudType>
const Foam::vector Foam::KinematicHWParcel<ParcelType>::calcVelocity
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt,
    const scalar Re,
    const scalar mu,
    const scalar mass,
    const vector& Su,
    vector& dUTrans,
    scalar& Spu
) const
{
    const typename TrackCloudType::parcelType& p =
        static_cast<const typename TrackCloudType::parcelType&>(*this);
    typename TrackCloudType::parcelType::trackingData& ttd =
        static_cast<typename TrackCloudType::parcelType::trackingData&>(td);

    const typename TrackCloudType::forceType& forces = cloud.forces();

    // Momentum source due to particle forces
    const forceSuSp Fcp = forces.calcCoupled(p, ttd, dt, mass, Re, mu);
    const forceSuSp Fncp = forces.calcNonCoupled(p, ttd, dt, mass, Re, mu);
    const scalar massEff = forces.massEff(p, ttd, mass);

    /*
    // Proper splitting ...
    // Calculate the integration coefficients
    const vector acp = (Fcp.Sp()*td.Uc() + Fcp.Su())/massEff;
    const vector ancp = (Fncp.Sp()*td.Uc() + Fncp.Su() + Su)/massEff;
    const scalar bcp = Fcp.Sp()/massEff;
    const scalar bncp = Fncp.Sp()/massEff;

    // Integrate to find the new parcel velocity
    const vector deltaUcp =
        cloud.UIntegrator().partialDelta
        (
            U_, dt, acp + ancp, bcp + bncp, acp, bcp
        );
    const vector deltaUncp =
        cloud.UIntegrator().partialDelta
        (
            U_, dt, acp + ancp, bcp + bncp, ancp, bncp
        );
    const vector deltaT = deltaUcp + deltaUncp;
    */

    // Shortcut splitting assuming no implicit non-coupled force ...
    vector Unew = Zero;
    if (p.stochastic() == false)
    {
        // Calculate the integration coefficients
        const vector acp = (Fcp.Sp()*td.Uc() + Fcp.Su())/massEff;
        const vector ancp = (Fncp.Su() + Su)/massEff;
        const scalar bcp = Fcp.Sp()/massEff;

        // Integrate to find the new parcel velocity
        const vector deltaU = cloud.UIntegrator().delta(U_, dt, acp + ancp, bcp);
        const vector deltaUncp = ancp*dt;
        const vector deltaUcp = deltaU - deltaUncp;

        // Calculate the new velocity and the momentum transfer terms
        Unew = U_ + deltaU;

        dUTrans -= massEff*deltaUcp;
        Spu = dt*Fcp.Sp();

    }
    else
    {
        // Rearrange integrated val to incorporate any resetting of U_ by patch interaction
        //const vector deltaU = p.UNext() - p.UN()*exp(-dt/max(p.tTurb(),VSMALL));
        

        //Keep as velocity from t = n, so that scheme is explicit
        Unew = U_; //*exp(-dt/max(p.tTurb(),VSMALL)) + deltaU ; //p.UNext();
    }
    // Apply correction to velocity and dUTrans for reduced-D cases
    const polyMesh& mesh = cloud.pMesh();
    meshTools::constrainDirection(mesh, mesh.solutionD(), Unew);
    meshTools::constrainDirection(mesh, mesh.solutionD(), dUTrans);

    return Unew;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const KinematicHWParcel<ParcelType>& p
)
:
    ParcelType(p),
    active_(p.active_),
    typeId_(p.typeId_),
    nParticle_(p.nParticle_),
    d_(p.d_),
    dTarget_(p.dTarget_),
    U_(p.U_),
    rho_(p.rho_),
    age_(p.age_),
    tTurb_(p.tTurb_),
    UTurb_(p.UTurb_)
{}


template<class ParcelType>
Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const KinematicHWParcel<ParcelType>& p,
    const polyMesh& mesh
)
:
    ParcelType(p, mesh),
    active_(p.active_),
    typeId_(p.typeId_),
    nParticle_(p.nParticle_),
    d_(p.d_),
    dTarget_(p.dTarget_),
    U_(p.U_),
    rho_(p.rho_),
    age_(p.age_),
    tTurb_(p.tTurb_),
    UTurb_(p.UTurb_)

{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
bool Foam::KinematicHWParcel<ParcelType>::move
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar trackTime
)
{
    typename TrackCloudType::parcelType& p =
        static_cast<typename TrackCloudType::parcelType&>(*this);
    typename TrackCloudType::parcelType::trackingData& ttd =
        static_cast<typename TrackCloudType::parcelType::trackingData&>(td);

    ttd.switchProcessor = false;
    ttd.keepParticle = true;

    const scalarField& cellLengthScale = cloud.cellLengthScale();
    const scalar maxCo = cloud.solution().maxCo();

    /*if (p.stochastic() == true)
    {
        this->U_ = p.UN();
    }*/

    while (ttd.keepParticle && !ttd.switchProcessor && p.stepFraction() < 1)
    {
        // Cache the current position, cell and step-fraction
        const point start = p.position();
        const scalar sfrac = p.stepFraction();
        
        // Total displacement over the time-step
        vector s = trackTime*U_; 
        // Cell length scale
        const scalar l = cellLengthScale[p.cell()];

        if (p.stochastic() == true)
        {
            //const scalar A = p.tTurb()*(1.-exp(-trackTime/p.tTurb())); //Stochastic integral coeff

            //Info << "compare s vs sStoc " << s << tab << (p.s()+A*this->U_) << endl;
            s = p.s();// + A*this->U_; //p.UN(); //p.U();

            scalar fTmp = 1 - p.stepFraction();
            fTmp = min(fTmp, maxCo);
            fTmp = min(fTmp, maxCo*l/max(small*l, mag(s)));

         /*   if (fTmp < 0.01)
            {
                Info << "breaking!" << endl;

                Info << "maxCo " << maxCo << tab 
                     << "maxCo*lmax(small*l, mag(s)) " << maxCo*l/max(small*l, mag(s)) << endl;
                Info << "sfrac " << sfrac << " stepFraction " << p.stepFraction() << endl;
                Info << "f " << fTmp << endl;
                Info << "displacement " << p.s() << endl;
                Info << "vel " << U_ << endl;
                Info << "dt " << (p.stepFraction() - sfrac)*trackTime << endl;
            }
*/
        }

        // Deviation from the mesh centre for reduced-D cases
        const vector d = p.deviationFromMeshCentre();

        // Fraction of the displacement to track in this loop. This is limited
        // to ensure that the both the time and distance tracked is less than
        // maxCo times the total value.
        scalar f = 1 - p.stepFraction();
        f = min(f, maxCo);
        f = min(f, maxCo*l/max(small*l, mag(s)));
        if (p.active())
        {
            // Track to the next face
            p.trackToFace(f*s - d, f);
        }
        else
        {
            // At present the only thing that sets active_ to false is a stick
            // wall interaction. We want the position of the particle to remain
            // the same relative to the face that it is on. The local
            // coordinates therefore do not change. We still advance in time and
            // perform the relevant interactions with the fixed particle.
            p.stepFraction() += f;
        }

        scalar dt = (p.stepFraction() - sfrac)*trackTime;
        // Avoid problems with extremely small timesteps
        if (dt > rootVSmall)
        {
            // Update cell based properties
            p.setCellValues(cloud, ttd);

            p.calcDispersion(cloud, ttd, dt);

            if (cloud.solution().cellValueSourceCorrection())
            {
                p.cellValueSourceCorrection(cloud, ttd, dt);
            }

            p.calc(cloud, ttd, dt);
        }
        p.age() += dt;
        if (p.onFace())
        {
            //Info << "onFace true" << endl;
            cloud.functions().postFace(p, ttd.keepParticle);
        }

        cloud.functions().postMove(p, dt, start, ttd.keepParticle);

        if (p.onFace() && ttd.keepParticle)
        {
            //Info << "hit face" << endl;
            p.hitFace(s, cloud, ttd);
        }
        /*Info << "Velocity is " << this-> U_ << " init vel was " << p.UN() << " predicted vel was " << p.UNext() << endl;
        Info << "displacement is " << s << endl;
        Info << "position is " << p.position() << endl;*/
    }

    // NOLONGER THIS//If stochastic parcel at start and end of timestep, no collision occured.
    // If velocity at end of sub-timestepping is same as UN (i.e. not perturbed by wall interaction)
    // Therefore can use UpNext from stochastic solver as p.U() for next averaging step
    //if (p.UN()==this->U_  )//p.stochastic() == true) 
    if (p.stochastic() == true)
    {
        this->U_ = p.UNext();
        p.U() = p.UNext();
    }

    return ttd.keepParticle;
}


template<class ParcelType>
template<class TrackCloudType>
bool Foam::KinematicHWParcel<ParcelType>::hitPatch
(
    TrackCloudType& cloud,
    trackingData& td
)
{
    typename TrackCloudType::parcelType& p =
        static_cast<typename TrackCloudType::parcelType&>(*this);

    const polyPatch& pp = p.mesh().boundaryMesh()[p.patch()];


    if (p.stochastic())
    {
        ;
    }
    else
    {
        // Invoke post-processing model
        cloud.functions().postPatch(p, pp, td.keepParticle);
    }
    // Invoke surface film model
    if (cloud.surfaceFilm().transferParcel(p, pp, td.keepParticle))
    {
        // All interactions done
        return true;
    }
    else if (pp.coupled())
    {
        // Don't apply the patchInteraction models to coupled boundaries
        return false;
    }
    else
    {
        // Invoke patch interaction model
        return cloud.patchInteraction().correct(p, pp, td.keepParticle);
    }
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::hitProcessorPatch
(
    TrackCloudType&,
    trackingData& td
)
{
    td.switchProcessor = true;
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::KinematicHWParcel<ParcelType>::hitWallPatch
(
    TrackCloudType&,
    trackingData&
)
{
    // wall interactions are handled by the generic hitPatch method
}


template<class ParcelType>
void Foam::KinematicHWParcel<ParcelType>::transformProperties(const tensor& T)
{
    ParcelType::transformProperties(T);

    U_ = transform(T, U_);
}


template<class ParcelType>
void Foam::KinematicHWParcel<ParcelType>::transformProperties
(
    const vector& separation
)
{
    ParcelType::transformProperties(separation);
}


// * * * * * * * * * * * * * * IOStream operators  * * * * * * * * * * * * * //

#include "KinematicHWParcelIO.C"

// ************************************************************************* //
