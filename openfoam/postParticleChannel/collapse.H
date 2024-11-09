    fileName path(UpTildeMean.rootPath()/UpTildeMean.caseName()/"graphs"/UpTildeMean.instance());
    mkDir(path);
    Info << tab << "Writing to" << path << endl;

    scalarField UMeanXvalues
    (
        channelIndexing.collapse(UMean.component(vector::X)())
    );

    scalarField UMeanYvalues
    (
        channelIndexing.collapse(UMean.component(vector::Y)())
    );

    scalarField UMeanZvalues
    (
        channelIndexing.collapse(UMean.component(vector::Z)())
    );

    //- get filtered particle velocity components
    scalarField UpTildeMeanXvalues
    (
        channelIndexing.collapse(UpTildeMean.component(vector::X)())
    );

    scalarField UpTildeMeanYvalues
    (
        channelIndexing.collapse(UpTildeMean.component(vector::Y)())
    );

    scalarField UpTildeMeanZvalues
    (
        channelIndexing.collapse(UpTildeMean.component(vector::Z)())
    );

    scalarField RxxValues(channelIndexing.collapse(Rxx));
    scalarField RyyValues(channelIndexing.collapse(Ryy));
    scalarField RzzValues(channelIndexing.collapse(Rzz));
    scalarField RxyValues(channelIndexing.collapse(Rxy, true));
    scalarField RxzValues(channelIndexing.collapse(Rxz, true));

    scalarField RpxxValues(channelIndexing.collapse(Rpxx));
    scalarField RpyyValues(channelIndexing.collapse(Rpyy));
    scalarField RpzzValues(channelIndexing.collapse(Rpzz));
    scalarField RpxyValues(channelIndexing.collapse(Rpxy, true));
    scalarField RpxzValues(channelIndexing.collapse(Rpxz, true));

    scalarField pPrime2MeanValues(channelIndexing.collapse(pPrime2Mean));

    scalarField urmsValues(sqrt(mag(RxxValues)));
    scalarField vrmsValues(sqrt(mag(RyyValues)));
    scalarField wrmsValues(sqrt(mag(RzzValues)));

    scalarField uprmsValues(sqrt(mag(RpxxValues)));
    scalarField vprmsValues(sqrt(mag(RpyyValues)));
    scalarField wprmsValues(sqrt(mag(RpzzValues)));

    scalarField kValues
    (
        0.5*(sqr(urmsValues) + sqr(vrmsValues) + sqr(wrmsValues))
    );

    //- probably need to change here to z
    const scalarField& y = channelIndexing.y();

    makeGraph(y, UMeanXvalues, "Uf", path, gFormat);
    makeGraph(y, UMeanZvalues, "Ufz", path, gFormat);
    makeGraph(y, urmsValues, "u", path, gFormat);
    makeGraph(y, vrmsValues, "v", path, gFormat);
    makeGraph(y, wrmsValues, "w", path, gFormat);
    makeGraph(y, RxyValues, "uv", path, gFormat);
    makeGraph(y, RxzValues, "uw", path, gFormat);
    makeGraph(y, kValues, "k", path, gFormat);

    if (UpTildeMeanHeader.typeHeaderOk<volVectorField>(true))
    {
        makeGraph(y, UpTildeMeanXvalues, "Upx", path, gFormat);
        makeGraph(y, UpTildeMeanZvalues, "Upz", path, gFormat);
    }
    if (UpTildePrime2MeanHeader.typeHeaderOk<volSymmTensorField>(true))
    {
        makeGraph(y, uprmsValues, "up", path, gFormat);
        makeGraph(y, vprmsValues, "vp", path, gFormat);
        makeGraph(y, wprmsValues, "wp", path, gFormat);
        makeGraph(y, RpxyValues, "uvp", path, gFormat);
        makeGraph(y, RpxzValues, "uwp", path, gFormat);
    }

    makeGraph(y, pPrime2MeanValues, "pPrime2Mean", path, gFormat);
