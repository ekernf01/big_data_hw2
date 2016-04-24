import pandas as pd
import ggplot as gg
import pickle as pkl
import math
from analysis.TestResult import TestResult
therunthatworked = "../results/LSH_vs_KDT_????? "
f = open(therunthatworked + ".pkl", 'rb')
results = pkl.load(f)
times =     [math.log(r.avg_time, 2)     for r in results]
distances = [r.avg_distance for r in results]
methods =   [r.method[0:3]  for r in results]
alpha =     [r.alpha  for r in results]
m =         [r.m  for r in results]
results_df = pd.DataFrame(data = {"times" : times,
                                  "distances" : distances,
                                  "methods" : methods,
                                  "m":m,
                                  "alpha": alpha})
print results_df
p = gg.ggplot(data = results_df, aesthetics = gg.aes(x = "times",
                                                     y = "distances",
                                                     label = "methods")) + \
    gg.geom_text() + \
    gg.ggtitle("LSH and KD trees: tradeoffs") + \
    gg.xlab("log2 average query time  ") + gg.ylab("Average L2 distance from query point)")
gg.ggsave(filename=therunthatworked + "log2.png", plot = p)