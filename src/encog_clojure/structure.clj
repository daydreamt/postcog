(ns encog-clojure.structure
  (:import [org.encog.engine.network.activation ActivationBiPolar
            ActivationClippedLinear ActivationCompetitive ActivationElliott
            ActivationElliottSymmetric ActivationGaussian ActivationLinear
            ActivationLOG ActivationRamp ActivationSigmoid ActivationSIN
            ActivationSoftMax ActivationStep ActivationSteepenedSigmoid
            ActivationTANH])
  (:use [clojure.tools.macro :only [macrolet]]))

(defrecord layer [neurons activation-function])
(defrecord connection [layer1 layer2 type])

(defn mklayer [n activation-function]
  "
  This creates a layer record, to pass to mknet later. It's sugar to ensure the
  function exists, but to use a different one, you would probably have to subclass
  org.encog.engine.network.activation.ActivationFunction.

  n is the number of neurons it should have

  The activation function must be one of the following
  :Bipolar
  :BipolarSteepenedSigmoid
  :ClippedLinear
  :Elliott
  :ElliottSymmetric
  :Gaussian
  :Log
  :Sin
  :Sigmoid
  :Softmax
  :SteepenedSigmoid
  :Tanh
  "
  (let [name-to-class (fn [a]
                        (eval `(new ~(symbol (str "Activation"
                                                  (apply str (rest (str a))))))))
        implemented (set [:Bipolar :BipolarSteepenedSigmoid :ClippedLinear
                          :Elliott :ElliottSymmetric :Gaussian :Log :Sin
                          :Sigmoid :Softmax :SteepenedSigmoid :Tanh])
        mappings {:Bipolar :BiPolar, :Log :LOG, :Sin :SIN, :Softmax :SoftMax}]
    (if (contains? implemented activation-function)
      ;;Great news, create the layer with a possibly renamed activation function
      (layer. n (name-to-class (get mappings activation-function activation-function)))
      (throw (Exception. (str "Error, class " activation-function " not found"))))))
