package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.bayesian.ParticleFilter;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.logit.fruehwirth.LogitFSWFFilter;
import plm.logit.fruehwirth.LogitMixParticle;
import plm.logit.fruehwirth.LogitParRBCWFFilter;
import plm.logit.fruehwirth.LogitRBCWFFilter;

import com.google.common.base.Preconditions;
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
      int N, int seed, int version) {

    ParticleFilter filter; 

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
    }

    filter.setNumParticles(N);

    final Matrix data = MatrixFactory.getDefault().copyArray(X);

    DataDistribution<LogitMixParticle> currentDist = (DataDistribution<LogitMixParticle>) filter.createInitialLearnedObject();
    final int Nx = data.getNumColumns();
    List<Double> logWeights = Lists.newArrayListWithExpectedSize(y.length*N);
    List<Double> stateMeans = Lists.newArrayListWithExpectedSize(y.length*N*Nx);
    
//    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
//    stateMeans[0] = 
//        Collections2.transform(currentDist.asMap().keySet(), 
//        new Function<LogitFSParticle, double[]>() {
//          @Override
//          public double[] apply(LogitFSParticle input) {
//            return input.getLinearState().getMean().toArray();
//          }}).toArray(new double[filter.getNumParticles()][data.getNumColumns()]);

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Matrix> obs = ObservedValue.<Vector, Matrix>create(
          t, 
          VectorFactory.getDefault().copyValues(y[t]),
          data.getSubMatrix(t, t, 0, data.getNumColumns()-1));
      filter.update(currentDist, obs);


//      logWeights[t] = new double[filter.getNumParticles()];
//      stateMeans[t] = new double[filter.getNumParticles()][data.getNumColumns()];
//      int k = 0;
      for (Entry<LogitMixParticle, ? extends Number> entry : currentDist.asMap().entrySet()) {
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
//          k++;
        }
      }
    }

    return new RLogitMixResult(stateMeans, logWeights);
  }
}
