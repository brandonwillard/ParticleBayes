package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.collection.ArrayUtil;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.VectorUtil;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import plm.gaussian.GaussianArHpWfParticle;
import plm.gaussian.GaussianArHpWfPlFilter;
import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.gaussian.GaussianArHpHmmPLFilter;
import plm.hmm.gaussian.GaussianArHpTransitionState;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.util.ObservedValue;

public class GaussianArHpPLAdapter {

  public static class GaussianArHpPLResult {

    private final double[] stateMeans;
    private final double[] stateCovs;
    private final double[] psiMeans;
    private final double[] psiCovs;
    private final double[] sigma2Shapes;
    private final double[] sigma2Scales;
    private final double[] logWeights;

    public GaussianArHpPLResult(List<Double> logWeights2,
        List<Double> stateMeans2, List<Double> stateCovs2, 
        List<Double> psiMeans2, List<Double> psiCovs2, 
        List<Double> sigma2Shapes2, List<Double> sigma2Scales2) {
      this.stateMeans = Doubles.toArray(stateMeans2);
      this.stateCovs = Doubles.toArray(stateCovs2);
      this.psiMeans = Doubles.toArray(psiMeans2);
      this.psiCovs = Doubles.toArray(psiCovs2);

      this.sigma2Shapes = Doubles.toArray(sigma2Shapes2);
      this.sigma2Scales = Doubles.toArray(sigma2Scales2);

      this.logWeights = Doubles.toArray(logWeights2);
    }

    public double[] getStateMeans() {
      return this.stateMeans;
    }

    public double[] getStateCovs() {
      return this.stateCovs;
    }

    public double[] getPsiMeans() {
      return this.psiMeans;
    }

    public double[] getPsiCovs() {
      return this.psiCovs;
    }

    public double[] getSigma2Shapes() {
      return this.sigma2Shapes;
    }

    public double[] getSigma2Scales() {
      return this.sigma2Scales;
    }

    public double[] getLogWeights() {
      return this.logWeights;
    }
    
  }


  /**
   * 
   * Note: initial sigma2 is the mean of the initial prior.
   * 
   * @param y
   * @param N
   * @param seed
   * @return
   */
  public static GaussianArHpPLResult batchUpdate(
      double[] initialStateMean, double[][] initialStateCov,
      double[] initialPsiMean, double[][] initialPsiCov,
      double initialSigma2Scale, double initialSigma2Shape,
      double[][] initialF, int K,
      double[][] y, int N, int seed) {
    
    final int Nx = initialStateMean.length;
    final int Ny = y[0].length;

    final Random rng = new Random(seed);

    final InverseGammaDistribution sigmaPrior = new InverseGammaDistribution(
        initialSigma2Shape, initialSigma2Scale);

    final double initialSigma2 = sigmaPrior.getMean();

    final int Np = initialPsiMean.length;
    final Vector psiMean = VectorFactory.getDefault().copyArray(initialPsiMean);
    final Matrix psiCov = MatrixFactory.getDefault().copyArray(initialPsiCov);
    final MultivariateGaussian phiPrior = new MultivariateGaussian(psiMean, psiCov);

    Matrix Ix = MatrixFactory.getDefault().createIdentity(Nx, Nx);
    Matrix Iy = MatrixFactory.getDefault().createIdentity(Ny, Ny);

    Matrix modelCovariance = Ix.scale(initialSigma2);
    Matrix measurementCovariance = Iy.scale(initialSigma2);
    
    Preconditions.checkArgument(
        psiMean.getDimensionality()/2 == initialStateMean.length);
    

    LinearDynamicalSystem dlm = new LinearDynamicalSystem(
        MatrixFactory.getDefault().createDiagonal(
           psiMean.subVector(Nx, 2*Nx - 1)),
        Ix.clone(),
        Ix.clone() 
      );
    KalmanFilter kf = new KalmanFilter(dlm, modelCovariance, measurementCovariance);
    kf.setCurrentInput(psiMean.subVector(0, Nx));

    final GaussianArHpWfPlFilter filter =
        new GaussianArHpWfPlFilter(kf, sigmaPrior, phiPrior, rng, K, true);
    filter.setNumParticles(N);

    DataDistribution<GaussianArHpWfParticle> currentDist = filter.createInitialLearnedObject();

    List<Double> logWeights = Lists.newArrayListWithCapacity(y.length * N);
    List<Double> stateMeans = Lists.newArrayListWithCapacity(y.length * N * Nx);
    List<Double> stateCovs = Lists.newArrayListWithCapacity(y.length * N * Nx*Nx);
    List<Double> psiMeans = Lists.newArrayListWithCapacity(y.length * N * Np);
    List<Double> psiCovs = Lists.newArrayListWithCapacity(y.length * N * Np*Np);
    List<Double> sigma2Shapes = Lists.newArrayListWithCapacity(y.length * N);
    List<Double> sigma2Scales = Lists.newArrayListWithCapacity(y.length * N);
    
//    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
//    int j = 0;
//    for (Entry<GaussianArHpWfParticle, ? extends Number> entry : 
//      currentDist.asMap().entrySet()) {
//      stateMeans[0][j] = entry.getKey().getState().getMean().toArray();
//      stateCovs[0][j] = entry.getKey().getState().getCovariance().toArray();
//      psiMeans[0][j] = entry.getKey().getPsiSS().getMean().toArray();
//      psiCovs[0][j] = entry.getKey().getPsiSS().getCovariance().toArray();
//      sigma2Shapes[0][j] = entry.getKey().getSigma2SS().getShape();
//      sigma2Scales[0][j] = entry.getKey().getSigma2SS().getScale();
//      j++;
//    }

    int z = 0;
    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Void> obs = ObservedValue.<Vector>create(
          t, 
          VectorFactory.getDefault().copyArray(y[t]));
      filter.update(currentDist, obs);

      for (Entry<GaussianArHpWfParticle, ? extends Number> entry : 
        currentDist.asMap().entrySet()) {
        final int count;
        if (entry.getValue() instanceof MutableDoubleCount) {
          count = ((MutableDoubleCount)entry.getValue()).count;
        } else {
          count = 1;
        }
        for (int r = 0; r < count; r++) {
          // FIXME might not be log scale
          final double normLogWeight = entry.getValue().doubleValue() - Math.log(count) - currentDist.getTotal();
          Preconditions.checkState(normLogWeight < 0d);
          logWeights.add(z, normLogWeight);
          Iterables.addAll(stateMeans, 
              Doubles.asList(entry.getKey().getState().getMean().toArray()));
          Iterables.addAll(stateCovs, 
              Doubles.asList(entry.getKey().getState().getCovariance().convertToVector().toArray()));
          Iterables.addAll(psiMeans, 
              Doubles.asList(entry.getKey().getPsiSS().getMean().toArray()));
          Iterables.addAll(psiCovs, 
              Doubles.asList(entry.getKey().getPsiSS().getCovariance().convertToVector().toArray()));
          sigma2Shapes.add(z, entry.getKey().getSigma2SS().getShape());
          sigma2Scales.add(z,  entry.getKey().getSigma2SS().getScale());
          z++;
        }
      }
    }

    return new GaussianArHpPLResult(logWeights, 
        stateMeans, stateCovs, 
        psiMeans, psiCovs, 
        sigma2Shapes, sigma2Scales);
  }
}
