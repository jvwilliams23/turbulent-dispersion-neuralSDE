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
//- initialise averaging objects 
AveragingMethod<vector>& upTildeAverage = upTildeAveragePtr();
AveragingMethod<scalar>& massAverageJW = massAverageJWPtr();

forAllIter(typename CloudType, this->owner(), iter)
{
    typename CloudType::parcelType& p = iter();
    const tetIndices tetIs(p.currentTetIndices());

    // can be used to set initial particle velocity equal to fluid
    if ((initCond_=="Uc") && (p.age()<2.5*deltaT))
    {
        p.U() = Uc[p.cell()]; //Us;
    }

    massAverageJW.add(p.coordinates(), tetIs, p.nParticle()*p.mass());
    upTildeAverage.add(p.coordinates(), tetIs, p.nParticle()*p.mass()*p.U());
}   
//- perform averaging
massAverageJW.average();
upTildeAverage.average(massAverageJW);
