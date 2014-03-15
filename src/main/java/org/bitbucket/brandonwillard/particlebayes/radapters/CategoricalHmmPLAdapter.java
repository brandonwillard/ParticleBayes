package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.learning.algorithm.hmm.HiddenMarkovModel;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.DefaultDataDistribution;

import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.hmm.HmmPlFilter;
import plm.hmm.HmmTransitionState;
import plm.hmm.StandardHMM;
import plm.hmm.categorical.CategoricalHmmPlFilter;
import plm.logit.fruehwirth.LogitMixParticle;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.util.ObservedValue;

public class CategoricalHmmPLAdapter {

  public static class CategoricalHMMPLResult {

    private final int[][] classIds;
    private final double[][] logWeights;

    public CategoricalHMMPLResult(int[][] classIds, double[][] logWeights) {
      this.classIds = classIds;
      this.logWeights = logWeights;
    }

    public int[][] getClassIds() {
      return this.classIds;
    }

    public double[][] getLogWeights() {
      return this.logWeights;
    }
    
  }


  /**
   * A HMM with discrete observations/responses/emissions.
   * 
   * @param hmmInitialProb
   * @param hmmTransitionProb
   * @param emissionProbs rows span components, columns span observation support 
   * @param y
   * @param N
   * @param resampleOnly 
   * @param seed
   * @return
   */
  public static CategoricalHMMPLResult batchUpdate(
      double[] hmmInitialProb, double[][] hmmTransitionProb, 
      double[][] emissionProbs,
      int[] y, int N, boolean resampleOnly, int seed) {
    
    final Random rng = new Random(seed);

    List<DefaultDataDistribution<Integer>> emissions = Lists.newArrayList();
    for (int i = 0; i < emissionProbs.length; i++) {
      DefaultDataDistribution<Integer> sLikelihood = new DefaultDataDistribution<Integer>();
      for (int j = 0; j < emissionProbs[i].length; j++) {
        sLikelihood.increment(j, emissionProbs[i][j]);
      }
      emissions.add(sLikelihood);
    }

    final StandardHMM<Integer> hmm =
        StandardHMM.create(
        new HiddenMarkovModel<Integer>(
            VectorFactory.getDefault().copyArray(hmmInitialProb),
            MatrixFactory.getDefault().copyArray(hmmTransitionProb), 
            emissions));

    final HmmPlFilter<StandardHMM<Integer>, HmmTransitionState<Integer, StandardHMM<Integer>>, Integer> filter =
        new CategoricalHmmPlFilter(hmm, rng, resampleOnly);
    filter.setNumParticles(N);

    DataDistribution<HmmTransitionState<Integer, StandardHMM<Integer>>> currentDist = filter.createInitialLearnedObject();
    double [][] logWeights = new double[y.length][filter.getNumParticles()];
    int [][] classIds = new int[y.length][filter.getNumParticles()];
    
//    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
//    classIds[0] = 
//        Ints.toArray(Collections2.transform(currentDist.asMap().keySet(), 
//        new Function<HmmTransitionState<Integer, StandardHMM<Integer>>, Integer>() {
//          @Override
//          public Integer apply(HmmTransitionState<Integer, StandardHMM<Integer>> input) {
//            return input.getClassId();
//          }}));

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Integer, Void> obs = ObservedValue.<Integer>create(
          t, y[t]);
      filter.update(currentDist, obs);

      logWeights[t] = new double[filter.getNumParticles()];
      classIds[t] = new int[filter.getNumParticles()];
      int k = 0;
      for (Entry<HmmTransitionState<Integer, StandardHMM<Integer>>, ? extends Number> entry : currentDist.asMap().entrySet()) {
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
          classIds[t][k] = entry.getKey().getClassId();
          k++;
        }
      }
    }

    return new CategoricalHMMPLResult(classIds, logWeights);
  }
}
