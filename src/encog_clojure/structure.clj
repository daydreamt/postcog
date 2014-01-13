(ns encog-clojure.structure
  (:import [org.encog.engine.network.activation ActivationBiPolar
            ActivationClippedLinear ActivationCompetitive ActivationElliott
            ActivationElliottSymmetric ActivationGaussian ActivationLinear
            ActivationLOG ActivationRamp ActivationSigmoid ActivationSIN
            ActivationSoftMax ActivationStep ActivationSteepenedSigmoid
            ActivationTANH])
  (:use [clojure.string :only [lower-case]]))


;;(contains? (map clojure.string/lower-case implemented) activation-function)
;;(get mappings activation-function activation-function)) type)

(defrecord layer [neurons activation-function type])
(defrecord connection [layer1 layer2 type bias recurrent?])

(defn mklayer [n activation-function & {:keys [type] :or {type :hidden}}]
  "
  This creates a layer record, to pass to mknet later. It's sugar to ensure the
  function exists, but to use a different one, you would probably have to subclass
  org.encog.engine.network.activation.ActivationFunction.

  n is the number of neurons it should have

  The optional parameter :type, should get one of the values :input, :hidden, or :output.
  Please set it for input and output, as ;;TODO: we can't autorecognize it yet.

  The activation function must be one of the following (it's caps insensitive)
  :Bipolar
  :BipolarSteepenedSigmoid
  :ClippedLinear
  :Elliott
  :ElliottSymmetric
  :Gaussian
  :Log
  :Linear
  :Sin
  :Sigmoid
  :Softmax
  :SteepenedSigmoid
  :Tanh
  "

  (let [name-to-class (fn [a]
                        (eval `(new ~(symbol (str "Activation"
                                                  (apply str (rest (str a))))))))
        activation-function (lower-case activation-function)
        implemented (set [:BiPolar :BipolarSteepenedSigmoid :ClippedLinear
                          :Elliott :ElliottSymmetric :Gaussian :Linear :LOG :SIN
                          :Sigmoid :SoftMax :SteepenedSigmoid :Tanh])
        matches (filter #(= activation-function (lower-case %)) implemented)
        mappings {:Bipolar :BiPolar, :Log :LOG, :Sin :SIN, :Softmax :SoftMax}]

    (if (not (empty? matches))
      ;;Great news, create the layer with a possibly renamed activation function
      (layer. n (name-to-class (first matches)) type)
      (throw (Exception. (str "Error, class " activation-function " not found"))))))


(defn mkconnection [source target & {:keys [bias recurrent?] :or
                                     {bias 1.0 recurrent? false type :full}}]
  "
  Makes a connection between the source and the target layers.
  Source and target must be layer records.
  Optional parameters:
  bias: the bias unit for that connection.
  TODO: Ask encog why it is limited to one unit per source connection.
  recurrent?: whether the connection should be recurrent or not.
  TODO: type is always :full for now, as it is not implemented in encog.
  "
  ;;TODO: Check types or something
  (connection. source target type bias recurrent?))


(defn mknet [layers connections]
  "
  Given seqs of layers and connections, creates a network.
  Please make sure at least one of the layer has type input,
  and at least one has type output, as encog will complain otherwise.
  "

  (let [inp (filter #(= :input (:type %)) layers)
        outp (filter #(= :output (:type %)) layers)
        hidden (filter f(= :hidden (:type %)) layers)]
    (assert (= 1 (count inp)) "Please give exactly one layer the type :input.")
    (assert (= 1 (count outp)) "Please give exactly one layer the type :output.")

  ))
