// correction velocity averages
    autoPtr<AveragingMethod<vector>> upTildeAveragePtr
    (
        AveragingMethod<vector>::New
        (
            IOobject
            (
                this->owner().name() + ":upTildeAverage",
                this->owner().db().time().timeName(),
                this->owner().mesh()
            ),
            this->owner().solution().dict(),
            this->owner().mesh()
        )
    );
    //- testing new mass average
    autoPtr<AveragingMethod<scalar>> massAverageJWPtr
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                this->owner().name() + ":massAverageJW",
                this->owner().db().time().timeName(),
                this->owner().mesh()
            ),
            this->owner().solution().dict(),
            this->owner().mesh()
        )
    );
    autoPtr<AveragingMethod<vector>> usTildeAveragePtr
    (
        AveragingMethod<vector>::New
        (
            IOobject
            (
                this->owner().name() + ":usTildeAverage",
                this->owner().db().time().timeName(),
                this->owner().mesh()
            ),
            this->owner().solution().dict(),
            this->owner().mesh()
        )
    );
    //-declare objects for filtered seen velocity for getting TKE
    // vector field of ones. 
    // Outer product of this and the vector of UsSqr diagonal components gives a tensor
    volVectorField ones
    (
        IOobject("ones",
                this->owner().mesh().time().timeName(),
                this->owner().mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
                ),
        this->owner().mesh(),
        dimensionedVector("0", dimless, vector(1.,1.,1.))
    );

    UcSeenInterpPtr_.reset
    (
        interpolation<vector>::New
        (
             this->owner().solution().interpolationSchemes(),
             Uc
        ).ptr()
    );


    //- initialise averaging objects 
    AveragingMethod<vector>& upTildeAverage = upTildeAveragePtr();
    AveragingMethod<scalar>& massAverageJW = massAverageJWPtr();
    AveragingMethod<vector>& usTildeAverage = usTildeAveragePtr();

    //  initialise a boolean that will print initial cond if particle age < 2.5 deltaT
    bool printInitialCond = false;

    if (stationary_)
    {   
        Info << "Particles are stationary" << endl;
    }

    forAllIter(typename CloudType, this->owner(), iter)
    {
        typename CloudType::parcelType& p = iter();
        const tetIndices tetIs(p.currentTetIndices());
        const label face = p.face();

        const vector Uc = UcSeenInterp().interpolate(p.coordinates(), tetIs, face);

        if (p.age()<2.5*deltaT)
        {
            if (initCond_!="0")
            {
                p.UTurb() = Uc;
                p.UTurbN() = p.UTurb();
            }
            if (initCond_=="Uc")
            {
                p.U() = Uc;
                p.UN() = p.U();
            }
            if (initCond_=="Up")
            {
                p.UN() = p.U();
            }
            if (initCond_=="UsIsUp")
            {
                // used for setting initial vel of LES same as DNS
                p.UN() = p.U();
                p.UTurb() = p.U();
                p.UTurbN() = p.U();
            }
            printInitialCond = true;
        }


        massAverageJW.add(p.coordinates(), tetIs, p.nParticle()*p.mass());
        upTildeAverage.add(p.coordinates(), tetIs, p.nParticle()*p.mass()*p.U());
        usTildeAverage.add(p.coordinates(), tetIs, p.nParticle()*p.mass()*p.UTurb());

    }
    //- perform averaging
    massAverageJW.average();
    upTildeAverage.average(massAverageJW);
    usTildeAverage.average(massAverageJW);

    UsTilde_.primitiveFieldRef() = usTildeAverage.primitiveField();//[0];
    UpTilde_.primitiveFieldRef() = upTildeAverage.primitiveField();//[0];

    volScalarField k = mesh.template
                 lookupObject<volScalarField>(kName_);;
    const volScalarField& epsilon = mesh.template
                 lookupObject<volScalarField>(epsilonName_);


    kInterpPtr_.reset
    (
        interpolation<scalar>::New
        (
            this->owner().solution().interpolationSchemes(),
            k
        ).ptr()
    );
    epsilonInterpPtr_.reset
    (
        interpolation<scalar>::New
        (
            this->owner().solution().interpolationSchemes(),
            epsilon
        ).ptr()
    );
    pGradInterpPtr_.reset
    (
         interpolation<vector>::New
         (
              this->owner().solution().interpolationSchemes(),
              gradPg
         ).ptr()
    );
    uLaplacianInterpPtr_.reset
    (
        interpolation<vector>::New
        (
             this->owner().solution().interpolationSchemes(),
             lapUc //viscStresses
        ).ptr()
    );
    uGradInterpPtr_.reset
    (
         interpolation<tensor>::New
         (
              this->owner().solution().interpolationSchemes(),
              gradUc
         ).ptr()
    );
    UpTildeInterpPtr_.reset
    (
        interpolation<vector>::New
        (
            this->owner().solution().interpolationSchemes(),
            UpTilde_
        ).ptr()
    );
    UsTildeInterpPtr_.reset
    (
        interpolation<vector>::New
        (
            this->owner().solution().interpolationSchemes(),
            UsTilde_
        ).ptr()
    );

    forAllIter(typename CloudType, this->owner(), iter)
    {
        typename CloudType::parcelType& p = iter();
        const tetIndices tetIs(p.currentTetIndices());
        const label face = p.face();

        p.UN() = p.U();
        p.UTurbN() = p.UTurb();
        p.UcTildeN() = UcSeenInterp().interpolate(p.coordinates(), tetIs, face);
        p.UpTildeN() = UpTildeInterp().interpolate(p.coordinates(), tetIs, face);
        p.UsTildeN() = UsTildeInterp().interpolate(p.coordinates(), tetIs, face);


        p.PGradTildeN() = pGradInterp().interpolate(p.coordinates(), tetIs, face);
        p.kN() = mag(kInterp().interpolate(p.coordinates(), tetIs, face));
        p.epsilonN() = mag(epsilonInterp().interpolate(p.coordinates(), tetIs, face));
        p.UcGradN() = uGradInterp().interpolate(p.coordinates(), tetIs, face);
        p.UcLaplacianN() = uLaplacianInterp().interpolate(p.coordinates(),tetIs, face);
        
        if (p.age()<2.5*deltaT)
        {
            if (initCond_ == "generatorTKE")
            {
                scalar uPrime = Cinit_ * sqrt(2.0/3.0 * mag(p.kN()));
                scalar randi = rnd.scalarNormal();
                scalar randj = rnd.scalarNormal();
                scalar randk = rnd.scalarNormal();
                vector G1i = vector(randi, randj, randk);
                p.UTurbN() = p.UcTildeN() + (uPrime * G1i);
                p.UN() = p.UTurbN();
                p.UTurb() = p.UTurbN();
                p.U() = p.UN();
                if (debugParticles_)
                {
                    Info << "generator output output " << uPrime << endl;
                    Info << "UcTilde " << p.UcTildeN() << endl;
                    Info << "UTurb " << p.UTurbN() << endl;
                    Info << "U " << p.UN() << endl;
                }
            }
        }

    }

