package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import plm.logit.LogitParticle;
import plm.logit.fruehwirth.LogitFSWFFilter;
import plm.logit.fruehwirth.LogitMixParticle;
import plm.logit.fruehwirth.LogitParRBCWFFilter;
import plm.logit.fruehwirth.LogitRBCWFFilter;
import plm.logit.polyagamma.LogitRBCPGWFFilter;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.util.ObservedValue;

public class LogitAdapter {

  public static class RLogitMixResult {

    private final double[] stateMeans;
    private final double[] logWeights;

    public RLogitMixResult(List<Double> stateMeans, List<Double> logWeights) {
      this.stateMeans = Doubles.toArray(stateMeans);
      this.logWeights = Doubles.toArray(logWeights);
    }

    public double[] getStateMeans() {
      return this.stateMeans;
    }

    public double[] getLogWeights() {
      return this.logWeights;
    }
    
  }

  public static RLogitMixResult batchUpdate(
      double[] y, double[][] X, 
      double[] priorMean, double[][] priorCov, 
      double[][] F, double[][] G, double[][] modelCov,
      int N, int K, int seed, int version) {

    ParticleFilter filter; 
    
    /*
     * TODO use the parallel filter if there are multiple cores
     * to spare...
     */
    if (version == 0) {
      version = Runtime.getRuntime().availableProcessors() > 1 ? 2 : 1;
    }

    if (version == 1) {
      // Uses suff. stats. in mixture likelihood and mixture utility sampling.
      filter = new LogitRBCWFFilter(
          new MultivariateGaussian(
              VectorFactory.getDefault().copyArray(priorMean),
              MatrixFactory.getDefault().copyArray(priorCov)), 
          MatrixFactory.getDefault().copyArray(F), 
          MatrixFactory.getDefault().copyArray(G), 
          MatrixFactory.getDefault().copyArray(modelCov), 
          new Random(seed));
      System.out.println("Running single-threaded RBC-WF mixture logit filter");
    } else if (version == 2) {
      // Uses suff. stats. in mixture likelihood and mixture utility sampling.
      // This one runs on multiple threads
      filter = new LogitParRBCWFFilter(
          new MultivariateGaussian(
              VectorFactory.getDefault().copyArray(priorMean),
              MatrixFactory.getDefault().copyArray(priorCov)), 
          MatrixFactory.getDefault().copyArray(F), 
          MatrixFactory.getDefault().copyArray(G), 
          MatrixFactory.getDefault().copyArray(modelCov), 
          new Random(seed));
      System.out.println("Running multi-threaded RBC-WF mixture logit filter");
    } else if (version == 3) {
      // This one uses the Polya-Gamma latent variable method
      filter = new LogitRBCPGWFFilter(
          new MultivariateGaussian(
              VectorFactory.getDefault().copyArray(priorMean),
              MatrixFactory.getDefault().copyArray(priorCov)), 
          MatrixFactory.getDefault().copyArray(F), 
          MatrixFactory.getDefault().copyArray(G), 
          MatrixFactory.getDefault().copyArray(modelCov), 
          K,
          new Random(seed));
      System.out.println("Running single-threaded RBC-WF Polya-Gamma logit filter");
    } else {
      // Samples beta for mixture likelihood and FS utility sampling.
      filter = new LogitFSWFFilter(
          new MultivariateGaussian(
              VectorFactory.getDefault().copyArray(priorMean),
              MatrixFactory.getDefault().copyArray(priorCov)), 
          MatrixFactory.getDefault().copyArray(F), 
          MatrixFactory.getDefault().copyArray(G), 
          MatrixFactory.getDefault().copyArray(modelCov), 
          new Random(seed));
      System.out.println("Running single-threaded FS-sampling mixture logit filter");
    }

    filter.setNumParticles(N);

    final Matrix data = MatrixFactory.getDefault().copyArray(X);

    DataDistribution<LogitParticle> currentDist = (DataDistribution<LogitParticle>) filter.createInitialLearnedObject();
    final int Nx = data.getNumColumns();
    List<Double> logWeights = Lists.newArrayListWithExpectedSize(y.length*N);
    List<Double> stateMeans = Lists.newArrayListWithExpectedSize(y.length*N*Nx);

    Stopwatch watch = new Stopwatch();
    UnivariateGaussian.SufficientStatistic latencyStats = 
        new UnivariateGaussian.SufficientStatistic();
    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Matrix> obs = ObservedValue.<Vector, Matrix>create(
          t, 
          VectorFactory.getDefault().copyValues(y[t]),
          data.getSubMatrix(t, t, 0, data.getNumColumns()-1));
      watch.reset();
      watch.start();
        filter.update(currentDist, obs);
      watch.stop();
      final long latency = watch.elapsed(TimeUnit.MILLISECONDS);
      latencyStats.update(latency);

      if ((t+1) % (y.length/20d) < 1) {
        System.out.println("t = " + t 
            + ", update latency mean=" + latencyStats.getMean());
        latencyStats.clear();
      }

      for (Entry<LogitParticle, ? extends Number> entry : currentDist.asMap().entrySet()) {
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
              Doubles.asList(entry.getKey().getLinearState().getMean().toArray()));
        }
      }
    }

    return new RLogitMixResult(stateMeans, logWeights);
  }
}
