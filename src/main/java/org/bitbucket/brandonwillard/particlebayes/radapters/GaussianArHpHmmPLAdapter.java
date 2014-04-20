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
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
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

public class GaussianArHpHmmPLAdapter {

  public static class GaussianArHpHmmPLResult {

    private final double[] stateMeans;
    private final double[] stateCovs;
    private final double[] psiMeans;
    private final double[] psiCovs;
    private final double[] sigma2Shapes;
    private final double[] sigma2Scales;
    private final double[] logWeights;

    public GaussianArHpHmmPLResult(List<Double> logWeights2,
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
   * Gaussian AR(1) HMM from ParticleLearningModels: {@link plm.hmm.gaussian.GaussianArHpHmmPLFilter}. 
   * The parameters (AR(1) and shared obs., state scale param.) vary across mixture components.
   * 
   * 
   * @param y
   * @param initialF
   * @param initialStateMean
   * @param initialStateCov
   * @param initialPsiMean
   * @param initialPsiCov
   * @param initialSigma2Scale
   * @param initialSigma2Shape
   * @param initialProbs
   * @param transProbs
   * @param K
   * @param N
   * @param seed
   * @return
   */
  public static GaussianArHpHmmPLResult batchUpdate(
      double[][] y, 
      double[][] initialF, 
      double[] initialStateMean, double[][] initialStateCov,
      double[] initialPsiMean, double[][] initialPsiCov,
      double initialSigma2Scale, double initialSigma2Shape,
      double[] initialProbs, double[][] transProbs,
      int K, int N, int seed) {
    
    // response dimension
    final int Ny = y[0].length;
    // state dimension
    final int Nx = initialStateMean.length;
    // this should simply be 2*Nx, no?
    final int Np = initialPsiMean.length;
    // number of hmm states
    final int M = initialProbs.length;
    Matrix Ix = MatrixFactory.getDefault().createIdentity(Nx, Nx);
    Matrix Iy = MatrixFactory.getDefault().createIdentity(Ny, Ny);

    final Random rng = new Random(seed);

    final InverseGammaDistribution sigmaPrior = new InverseGammaDistribution(
        initialSigma2Shape, initialSigma2Scale);
    final double initialSigma2 = sigmaPrior.getMean();
    Matrix modelCovariance = Ix.scale(initialSigma2);
    Matrix measurementCovariance = Iy.scale(initialSigma2);

    List<KalmanFilter> kfs = Lists.newArrayList();
    List<MultivariateGaussian> priorPhis = Lists.newArrayList();
    for (int i = 0; i < initialProbs.length; i++) {

      final Vector psiMean = VectorFactory.getDefault().copyArray(initialPsiMean);
      final Matrix psiCov = MatrixFactory.getDefault().copyArray(initialPsiCov);

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
      kfs.add(kf);
      final MultivariateGaussian priorPhi = new MultivariateGaussian(psiMean, psiCov);
      priorPhis.add(priorPhi);
    }

    Vector initialClassProbs = VectorFactory.getDefault()
            .copyArray(initialProbs);
    Matrix classTransProbs = MatrixFactory.getDefault().copyArray(transProbs);
    
    DlmHiddenMarkovModel hmm = new DlmHiddenMarkovModel(
        kfs, initialClassProbs, classTransProbs);

    final HmmPlFilter<DlmHiddenMarkovModel, GaussianArHpTransitionState, Vector> filter =
        new GaussianArHpHmmPLFilter(hmm, sigmaPrior, priorPhis, rng, true);
    filter.setNumParticles(N);

    DataDistribution<GaussianArHpTransitionState> currentDist = filter.createInitialLearnedObject();

    List<Double> logWeights = Lists.newArrayListWithCapacity(y.length * N);
    List<Double> stateMeans = Lists.newArrayListWithCapacity(y.length * N * Nx);
    List<Double> stateCovs = Lists.newArrayListWithCapacity(y.length * N * Nx*Nx);
    List<Double> psiMeans = Lists.newArrayListWithCapacity(y.length * N * Np * M);
    List<Double> psiCovs = Lists.newArrayListWithCapacity(y.length * N * Np*Np * M);
    List<Double> sigma2Shapes = Lists.newArrayListWithCapacity(y.length * N * M);
    List<Double> sigma2Scales = Lists.newArrayListWithCapacity(y.length * N * M);
    
    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Void> obs = ObservedValue.<Vector>create(
          t, 
          VectorFactory.getDefault().copyArray(y[t]));
      filter.update(currentDist, obs);

      for (Entry<GaussianArHpTransitionState, ? extends Number> entry : 
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
          logWeights.add(normLogWeight);
            Iterables.addAll(stateMeans, 
                Doubles.asList(entry.getKey().getState().getMean().toArray()));
            Iterables.addAll(stateCovs, 
                Doubles.asList(entry.getKey().getState().getCovariance().convertToVector().toArray()));
          for (int i = 0; i < M; i++) {
            Iterables.addAll(psiMeans, 
                Doubles.asList(entry.getKey().getPsiSS().get(i).getMean().toArray()));
            Iterables.addAll(psiCovs, 
                Doubles.asList(entry.getKey().getPsiSS().get(i).getCovariance().convertToVector().toArray()));
            sigma2Shapes.add(entry.getKey().getInvScaleSS().getShape());
            sigma2Scales.add(entry.getKey().getInvScaleSS().getScale());
          }
        }
      }
    }

    return new GaussianArHpHmmPLResult(logWeights, 
        stateMeans, stateCovs, 
        psiMeans, psiCovs, 
        sigma2Shapes, sigma2Scales);
  }
}
