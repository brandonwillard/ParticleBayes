package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.signals.LinearDynamicalSystem;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.KalmanFilter;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;
import gov.sandia.cognition.statistics.distribution.InverseGammaDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.gaussian.GaussianArHpWfParticle;
import plm.gaussian.GaussianArHpWfPlFilter;
import plm.hmm.DlmHiddenMarkovModel;
import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.gaussian.GaussianArHpHmmPlFilter;
import plm.hmm.gaussian.GaussianArHpTransitionState;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.util.ObservedValue;

public class GaussianArHpPLAdapter {

  public static class GaussianArHpPLResult {

    private final double[][][] stateMeans;
    private final double[][][][] stateCovs;
    private final double[][][] psiMeans;
    private final double[][][][] psiCovs;
    private final double[][] sigma2Shapes;
    private final double[][] sigma2Scales;
    private final double[][] logWeights;

    public GaussianArHpPLResult(double[][] logWeights,
        double[][][] stateMeans, double[][][][] stateCovs, 
        double[][][] psiMeans, double[][][][] psiCovs, 
        double [][] covShapes, double[][] covScales) {
      this.stateMeans = stateMeans;
      this.stateCovs = stateCovs;
      this.psiMeans = psiMeans;
      this.psiCovs = psiCovs;

      this.sigma2Shapes = covShapes;
      this.sigma2Scales = covScales;

      this.logWeights = logWeights;
    }

    public double[][][] getStateMeans() {
      return this.stateMeans;
    }

    public double[][][][] getStateCovs() {
      return this.stateCovs;
    }

    public double[][][] getPsiMeans() {
      return this.psiMeans;
    }

    public double[][][][] getPsiCovs() {
      return this.psiCovs;
    }

    public double[][] getSigma2Shapes() {
      return this.sigma2Shapes;
    }

    public double[][] getSigma2Scales() {
      return this.sigma2Scales;
    }

    public double[][] getLogWeights() {
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
    double [][] logWeights = new double[y.length][N];
    double [][][] stateMeans = new double[y.length][N][];
    double [][][][] stateCovs = new double[y.length][N][][];
    double [][][] psiMeans = new double[y.length][N][];
    double [][][][] psiCovs = new double[y.length][N][][];
    double [][] sigma2Shapes = new double[y.length][N];
    double [][] sigma2Scales = new double[y.length][N];
    
    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
    int j = 0;
    for (Entry<GaussianArHpWfParticle, ? extends Number> entry : 
      currentDist.asMap().entrySet()) {
      stateMeans[0][j] = entry.getKey().getState().getMean().toArray();
      stateCovs[0][j] = entry.getKey().getState().getCovariance().toArray();
      psiMeans[0][j] = entry.getKey().getPsiSS().getMean().toArray();
      psiCovs[0][j] = entry.getKey().getPsiSS().getCovariance().toArray();
      sigma2Shapes[0][j] = entry.getKey().getSigma2SS().getShape();
      sigma2Scales[0][j] = entry.getKey().getSigma2SS().getScale();
      j++;
    }

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Void> obs = ObservedValue.<Vector>create(
          t, 
          VectorFactory.getDefault().copyArray(y[t]));
      filter.update(currentDist, obs);

      logWeights[t] = new double[filter.getNumParticles()];
      int k = 0;
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
          logWeights[t][k] = normLogWeight;
          stateMeans[t][k] = entry.getKey().getState().getMean().toArray();
          stateCovs[t][k] = entry.getKey().getState().getCovariance().toArray();
          psiMeans[t][k] = entry.getKey().getPsiSS().getMean().toArray();
          psiCovs[t][k] = entry.getKey().getPsiSS().getCovariance().toArray();
          sigma2Shapes[t][k] = entry.getKey().getSigma2SS().getShape();
          sigma2Scales[t][k] = entry.getKey().getSigma2SS().getScale();
          k++;
        }
      }
    }

    return new GaussianArHpPLResult(logWeights, 
        stateMeans, stateCovs, 
        psiMeans, psiCovs, 
        sigma2Shapes, sigma2Scales);
  }
}
