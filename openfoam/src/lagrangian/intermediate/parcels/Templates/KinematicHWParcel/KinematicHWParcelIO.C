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
#include "IOstreams.H"
#include "IOField.H"
#include "Cloud.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::KinematicHWParcel<ParcelType>::propertyList_ =
    Foam::KinematicHWParcel<ParcelType>::propertyList();

template<class ParcelType>
const std::size_t Foam::KinematicHWParcel<ParcelType>::sizeofFields_
(
    sizeof(KinematicHWParcel<ParcelType>)
  - offsetof(KinematicHWParcel<ParcelType>, active_)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::KinematicHWParcel<ParcelType>::KinematicHWParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields
)
:
    ParcelType(mesh, is, readFields),
    active_(false),
    typeId_(0),
    nParticle_(0.0),
    d_(0.0),
    dTarget_(0.0),
    U_(Zero),
    rho_(0.0),
    age_(0.0),
    tTurb_(0.0),
    UTurb_(Zero),
    UTurbN_(Zero),
    UN_(Zero),
    UNext_(Zero),
    UcTildeN_(Zero), // JW 20/09/21
    kN_(0.0), // JW 16/09/21
    epsilonN_(0.0), // JW 16/09/21
    UcGradN_(Zero)
{
    if (readFields)
    {
        if (is.format() == IOstream::ASCII)
        {
            active_ = readBool(is);
            typeId_ = readLabel(is);
            nParticle_ = readScalar(is);
            d_ = readScalar(is);
            dTarget_ = readScalar(is);
            is >> U_;
            rho_ = readScalar(is);
            age_ = readScalar(is);
            tTurb_ = readScalar(is);
            is >> UTurb_;
            is >> UcTildeN_; // JW 20/09/21
            is >> UcGradN_;
            kN_ = readScalar(is); // JW 16/09/21
            epsilonN_ = readScalar(is); // JW 16/09/21
            is >> UNext_;
            is >> UTurbN_;
            is >> UN_;
        }
        else
        {
            is.read(reinterpret_cast<char*>(&active_), sizeofFields_);
        }
    }

    // Check state of Istream
    is.check
    (
        "KinematicHWParcel<ParcelType>::KinematicHWParcel"
        "(const polyMesh&, Istream&, bool)"
    );
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicHWParcel<ParcelType>::readFields(CloudType& c)
{
    bool valid = c.size();

    ParcelType::readFields(c);

    IOField<label> active
    (
        c.fieldIOobject("active", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, active);

    IOField<label> typeId
    (
        c.fieldIOobject("typeId", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, typeId);

    IOField<scalar> nParticle
    (
        c.fieldIOobject("nParticle", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, nParticle);

    IOField<scalar> d
    (
        c.fieldIOobject("d", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, d);

    IOField<scalar> dTarget
    (
        c.fieldIOobject("dTarget", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, dTarget);

    IOField<vector> U
    (
        c.fieldIOobject("U", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, U);

    IOField<scalar> rho
    (
        c.fieldIOobject("rho", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, rho);

    IOField<scalar> age
    (
        c.fieldIOobject("age", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, age);

    IOField<scalar> tTurb
    (
        c.fieldIOobject("tTurb", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, tTurb);

    IOField<vector> UTurb
    (
        c.fieldIOobject("UTurb", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, UTurb);
    // JW 20/09/21
    IOField<vector> UcTildeN
    (
        c.fieldIOobject("Uc", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, UcTildeN);
    IOField<vector> UTurbN
    (
        c.fieldIOobject("UTurbN", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, UTurbN);
    IOField<vector> UN
    (
        c.fieldIOobject("UN", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, UN);

    IOField<vector> UNext
    (
        c.fieldIOobject("UNext", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, UNext);

    IOField<tensor> UcGradN
    (
        c.fieldIOobject("UcGradN", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, UcGradN);

    // JW 16/09/21
    IOField<scalar> epsilonN
    (
        c.fieldIOobject("epsilon", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, epsilonN);
    IOField<scalar> kN
    (
        c.fieldIOobject("k", IOobject::READ_IF_PRESENT),
        valid
    );
    c.checkFieldIOobject(c, kN);

    label i = 0;

    forAllIter(typename CloudType, c, iter)
    {
        KinematicHWParcel<ParcelType>& p = iter();

        p.active_ = active[i];
        p.typeId_ = typeId[i];
        p.nParticle_ = nParticle[i];
        p.d_ = d[i];
        p.dTarget_ = dTarget[i];
        p.U_ = U[i];
        p.rho_ = rho[i];
        p.age_ = age[i];
        p.tTurb_ = tTurb[i];
        p.UTurb_ = UTurb[i];
        p.UcTildeN_ = UcTildeN[i];  // JW 20/09/21
        p.UTurbN_ = UTurbN[i];
        p.UN_ = UN[i];
        p.UNext_ = UNext[i];
        p.UcGradN_ = UcGradN[i];
        p.epsilonN_ = epsilonN[i]; // JW 16/09/21
        p.kN_ = kN[i]; // JW 16/09/21
        i++;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicHWParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);

    label np = c.size();

    IOField<label> active(c.fieldIOobject("active", IOobject::NO_READ), np);
    IOField<label> typeId(c.fieldIOobject("typeId", IOobject::NO_READ), np);
    IOField<scalar> nParticle
    (
        c.fieldIOobject("nParticle", IOobject::NO_READ),
        np
    );
    IOField<scalar> d(c.fieldIOobject("d", IOobject::NO_READ), np);
    IOField<scalar> dTarget(c.fieldIOobject("dTarget", IOobject::NO_READ), np);
    IOField<vector> U(c.fieldIOobject("U", IOobject::NO_READ), np);
    IOField<scalar> rho(c.fieldIOobject("rho", IOobject::NO_READ), np);
    IOField<scalar> age(c.fieldIOobject("age", IOobject::NO_READ), np);
    IOField<scalar> tTurb(c.fieldIOobject("tTurb", IOobject::NO_READ), np);
    IOField<vector> UTurb(c.fieldIOobject("UTurb", IOobject::NO_READ), np);
    // JW 20/09/21
    IOField<vector> UcTildeN(c.fieldIOobject("Uc", IOobject::NO_READ), np);
    IOField<tensor> UcGradN(c.fieldIOobject("UcGradN", IOobject::NO_READ), np);
    IOField<vector> UTurbN(c.fieldIOobject("UTurbN", IOobject::NO_READ), np);
    IOField<vector> UN(c.fieldIOobject("UN", IOobject::NO_READ), np);
    IOField<vector> UNext(c.fieldIOobject("UNext", IOobject::NO_READ), np);

    // JW 16/09/21
    IOField<scalar> epsilonN(c.fieldIOobject("epsilon", IOobject::NO_READ), np);
    IOField<scalar> kN(c.fieldIOobject("k", IOobject::NO_READ), np);

    label i = 0;

    forAllConstIter(typename CloudType, c, iter)
    {
        const KinematicHWParcel<ParcelType>& p = iter();

        active[i] = p.active();
        typeId[i] = p.typeId();
        nParticle[i] = p.nParticle();
        d[i] = p.d();
        dTarget[i] = p.dTarget();
        U[i] = p.U();
        rho[i] = p.rho();
        age[i] = p.age();
        tTurb[i] = p.tTurb();
        UTurb[i] = p.UTurb();
        UcTildeN[i] = p.UcTildeN(); // JW 20/09/21
        UcGradN[i] = p.UcGradN();
        UTurbN[i] = p.UTurbN();
        UN[i] = p.UN();
        UNext[i] = p.UNext();
        // JW 16/09/21
        epsilonN[i] = p.epsilonN();
        kN[i] = p.kN();

        i++;
    }

    const bool valid = np > 0;

    active.write(valid);
    typeId.write(valid);
    nParticle.write(valid);
    d.write(valid);
    dTarget.write(valid);
    U.write(valid);
    rho.write(valid);
    age.write(valid);
    tTurb.write(valid);
    UTurb.write(valid);
    UcTildeN.write(valid);  // JW 20/09/21
    UcGradN.write(valid);
    UTurbN.write(valid);
    UN.write(valid);
    UNext.write(valid);
    // JW 16/09/21
    epsilonN.write(valid);
    kN.write(valid);
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const KinematicHWParcel<ParcelType>& p
)
{
    if (os.format() == IOstream::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << p.active()
            << token::SPACE << p.typeId()
            << token::SPACE << p.nParticle()
            << token::SPACE << p.d()
            << token::SPACE << p.dTarget()
            << token::SPACE << p.U()
            << token::SPACE << p.rho()
            << token::SPACE << p.age()
            << token::SPACE << p.tTurb()
            << token::SPACE << p.UTurb()
            << token::SPACE << p.UcTildeN()  // JW 20/09/21
            << token::SPACE << p.UcGradN()
            << token::SPACE << p.UTurbN()
            << token::SPACE << p.UN()
            << token::SPACE << p.UNext()
            // JW 16/09/21
            << token::SPACE << p.epsilonN()
            << token::SPACE << p.kN()
            ;

    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.active_),
            KinematicHWParcel<ParcelType>::sizeofFields_
        );
    }

    // Check state of Ostream
    os.check
    (
        "Ostream& operator<<(Ostream&, const KinematicHWParcel<ParcelType>&)"
    );

    return os;
}


// ************************************************************************* //
