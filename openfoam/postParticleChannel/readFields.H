    // call headers for each field to check they exist later
    IOobject UMeanHeader
    (
        UName+"Mean",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ
    );
    IOobject alphaMeanHeader
    (
        alphaName+"Mean",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ
    );
    IOobject UpTildeMeanHeader
    (
        "UpTildeMean",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ
    );
    IOobject UpTildePrime2MeanHeader
    (
        "UpTildePrime2Mean",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ
    );

    if (!UMeanHeader.typeHeaderOk<volVectorField>(true))
    {
        Info<< "    No UMean field" << endl;
        continue;
    }

    if (!alphaMeanHeader.typeHeaderOk<volScalarField>(true))
    {
	Info<< "    No alphaMean field" << endl;	
    }

    volVectorField UpTildeMean
    (
	    UpTildeMeanHeader,
	    mesh
    );

    volSymmTensorField UpTildePrime2Mean
    (
	    UpTildePrime2MeanHeader,
        mesh
    );
    
    volVectorField UMean
    (
        UMeanHeader,
        mesh
    );
    volSymmTensorField UPrime2Mean
    (
        IOobject
        (
            UName+"Prime2Mean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    volScalarField Rxx(UPrime2Mean.component(symmTensor::XX));
    volScalarField Ryy(UPrime2Mean.component(symmTensor::YY));
    volScalarField Rzz(UPrime2Mean.component(symmTensor::ZZ));
    volScalarField Rxy(UPrime2Mean.component(symmTensor::XY));
    volScalarField Rxz(UPrime2Mean.component(symmTensor::XZ));


    if (UpTildeMeanHeader.typeHeaderOk<volVectorField>(true))
    {
        volVectorField UpTildeMeanNew
        (
             UpTildeMeanHeader,
             mesh	
        );
        UpTildeMean = UpTildeMeanNew;
    }
    else
    {
	Info<< "    No UpTildeMean field" << endl;
    }

    if (UpTildePrime2MeanHeader.typeHeaderOk<volSymmTensorField>(true))
    {
        volSymmTensorField UpTildePrime2MeanNew
        (
            UpTildePrime2MeanHeader,
            mesh
        );
        UpTildePrime2Mean = UpTildePrime2MeanNew;
    }
    else
    {
        Info<< "    No UpTildePrime2Mean field" << endl;
    }

/*    volVectorField UpTildeMean
    (
        IOobject
        (
            "UpTildeMean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    volSymmTensorField UpTildePrime2Mean
    (
        IOobject
        (
            "UpTildePrime2Mean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );*/
    volScalarField Rpxx(UpTildePrime2Mean.component(symmTensor::XX));
    volScalarField Rpyy(UpTildePrime2Mean.component(symmTensor::YY));
    volScalarField Rpzz(UpTildePrime2Mean.component(symmTensor::ZZ));
    volScalarField Rpxy(UpTildePrime2Mean.component(symmTensor::XY));
    volScalarField Rpxz(UpTildePrime2Mean.component(symmTensor::XZ));

    volScalarField pPrime2Mean
    (
        IOobject
        (
            "pPrime2Mean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );

    /*
    volScalarField epsilonMean
    (
        IOobject
        (
            "epsilonMean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );

    volScalarField nuMean
    (
        IOobject
        (
            "nuMean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );

    volScalarField gammaDotMean
    (
        IOobject
        (
            "gammaDotMean",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );

    volScalarField nuPrime2
    (
        IOobject
        (
            "nuPrime",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    volScalarField nuPrime = sqrt(mag(nuPrime2 - sqr(nuMean)));


    volScalarField gammaDotPrime2
    (
        IOobject
        (
            "gammaDotPrime",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    volScalarField gammaDotPrime = sqrt(mag(gammaDotPrime2 -sqr(gammaDotMean)));
    */
