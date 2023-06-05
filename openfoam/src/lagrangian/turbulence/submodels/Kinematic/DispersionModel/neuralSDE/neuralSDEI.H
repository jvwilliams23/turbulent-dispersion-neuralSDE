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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::UcSeenInterp() const
{
    if (!UcSeenInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Carrier phase Uc interpolation object not set"
            << abort(FatalError);
    }

    return UcSeenInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::pGradInterp() const
{
    if (!pGradInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Carrier phase pGrad interpolation object not set"
            << abort(FatalError);
    }

    return pGradInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::tensor>&
Foam::neuralSDE<CloudType>::uGradInterp() const
{
    if (!uGradInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Carrier phase velocity gradient interpolation object not set"
            << abort(FatalError);
    }

    return uGradInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::UpTildeInterp() const
{
    if (!UpTildeInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Particle phase velocity interpolation object not set"
            << abort(FatalError);
    }

    return UpTildeInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::UsTildeInterp() const
{
    if (!UsTildeInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Particle phase velocity seen interpolation object not set"
            << abort(FatalError);
    }

    return UsTildeInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::UpPredTildeInterp() const
{
    if (!UpPredTildeInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Predicted particle phase velocity interpolation object not set"
            << abort(FatalError);
    }

    return UpPredTildeInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::UsPredTildeInterp() const
{
    if (!UsPredTildeInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Predicted particle phase velocity seen interpolation object not set"
            << abort(FatalError);
    }

    return UsPredTildeInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::vector>&
Foam::neuralSDE<CloudType>::uLaplacianInterp() const
{
    if (!uLaplacianInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "Carrier phase velocity laplacian interpolation object not set"
            << abort(FatalError);
    }

    return uLaplacianInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::scalar>&
Foam::neuralSDE<CloudType>::kInterp() const
{
    if (!kInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "k interpolation object not set"
            << abort(FatalError);
    }

    return kInterpPtr_();
}

template<class CloudType>
inline const Foam::interpolation<Foam::scalar>&
Foam::neuralSDE<CloudType>::epsilonInterp() const
{
    if (!epsilonInterpPtr_.valid())
    {
        FatalErrorInFunction
            << "epsilon interpolation object not set"
            << abort(FatalError);
    }

    return epsilonInterpPtr_();
}

// ************************************************************************* //
