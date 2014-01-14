(ns encog-clojure.core
  (:import [org.encog.ml.data MLDataSet]
           [org.encog.ml.data.basic BasicMLData BasicMLDataSet]
           [org.encog.neural.freeform.training FreeformResilientPropagation FreeformBackPropagation]
           [org.encog.util.simple EncogUtility])
  (:use [encog-clojure.structure])

  (:use
   [clojure.core.matrix]
   [clojure.core.matrix.operators]))

(set-current-implementation :clatrix)  ;; Because vectorz, uhm, fails

;;Example to make it
;;XOR data as vectors
(def inp [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])
(def outp [[0.0] [1.0] [1.0] [0.0]])

(defn- to-double-array [xs]
  (into-array (map double-array xs)))


(defn- ds
  "Makes a dataset usable from encog for various? input types.
  TODO: This is probably not a good idea, as different algorithms need different
  datatype inputs. Let the overloading happen there."
  ([inp targets] (BasicMLDataSet. (to-double-array inp) (to-double-array targets)))
  ([inp] (BasicMLData. (double-array inp))))

(defn classify [nn inp]
  "Given input as a vector or core.matrix vector, return classified result from the net"
  (. nn (classify (ds inp))))

(defn compute [nn inp]
  "Have the network nn do a forward pass on the input inp.
  inp: clojure or core.matrix vector
  "
  (. nn (compute (ds inp))))

(defn train [nn inp output & {:keys [error lr momentum iterations algo]
                              :or {error 0.01 lr 0.7 momentum 0.3 iterations 10 algo :backprop}}]
  "TODO: Decide on the parameters.
   TODO: Generalize for other learning algorithms, etc.
   I think maybe you should keep trainers after all? It's one function call less. I guess.
   Or have them be records of some sort. Meh.
  "
  (let [data (ds inp output)
        trainer (FreeformResilientPropagation. nn data)]
    (doseq [i (range iterations)]
      (println "Iteration " i (.getError trainer)))))
