
find.log.alpha <- function(rLogWeights, rN) {
  suO = J("plm.utils.SamplingUtils")
  logWeights = as.double(rLogWeights)
  N = as.integer(rN)
  result = suO$findLogAlpha(logWeights, N)
  
  return(result)
}


water.filling.resample <- function(rLogWeights, rLogSum, rObjects, rN, seed=NULL) {
  wfrO = J("plm.utils.SamplingUtils")
  rng = new(J("java.util.Random"))
  if (!is.null(seed))
    rng$setSeed(seed)
  objects = new(J("java.util.ArrayList"))
  for (obj in rObjects) {
    objects$add(as.character(obj))
  }
      #J("com.google.common.primitives.Doubles")$asList(as.double(rObjects)))
  logWeights = as.double(rLogWeights)
  N = as.integer(rN)
  logSum = as.double(rLogSum)

  result = wfrO$waterFillingResample(logWeights, logSum, objects, rng, N)
  
  hm = result$asMap()
  # convert to R list
  keySet = .jrcall(hm,"keySet")
  an_iter = .jrcall(keySet,"iterator")
  aList = list()
  while(.jrcall(an_iter,"hasNext")) {
      key = .jrcall(an_iter,"next", simplify=F);
      skey = as.character(.jsimplify(key))
      val = .jrcall(hm,"get",key)
      if (.jinstanceof(val, "plm.utils.MutableDoubleCount")) {
        aList[[skey]]$count = val$getCount() 
        aList[[skey]]$value = val$getValue() 
      } else {
        aList[[skey]] = val$getValue()
      }
  }
  
  return(aList)
}

