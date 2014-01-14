(ns encog-clojure.structure
  (:import [org.encog.engine.network.activation ActivationBiPolar
            ActivationClippedLinear ActivationCompetitive ActivationElliott
            ActivationElliottSymmetric ActivationGaussian ActivationLinear
            ActivationLOG ActivationRamp ActivationSigmoid ActivationSIN
            ActivationSoftMax ActivationStep ActivationSteepenedSigmoid
            ActivationTANH]
           [org.encog.neural.freeform FreeformLayer FreeformNetwork])
  (:use [clojure.string :only [lower-case]]))

;; I have to assign ids to layers. Big hashes might be nicer, this will do.
(def ^{:private true} counter (atom java.math.BigInteger/ONE))
(defn- get-unique-id! []
  "Gets a unique id from the counter, then increments it."
  (let [res @counter]
    (swap! counter inc)))


(defrecord layer [id neurons activation-function type])
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
                        (eval `(new ~(symbol (str "org.encog.engine.network"
                                                  ".activation.Activation"
                                                  (apply str (rest (str a))))))))
        activation-function (lower-case activation-function)
        implemented (set [:BiPolar :BipolarSteepenedSigmoid :ClippedLinear
                          :Elliott :ElliottSymmetric :Gaussian :Linear :LOG :SIN
                          :Sigmoid :SoftMax :SteepenedSigmoid :TANH])
        matches (filter #(= activation-function (lower-case %)) implemented)
        mappings {:Bipolar :BiPolar, :Log :LOG, :Sin :SIN, :Softmax :SoftMax}]

    (if (not (empty? matches))
      ;;Great news, create the layer with a possibly renamed activation function
      (layer. (get-unique-id!) n (name-to-class (first matches)) type)
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
  TODO: Encog ignores recurrent? What a mess.
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
        hidden (filter #(= :hidden (:type %)) layers)]
    (assert (= 1 (count inp)) "Please give exactly one layer the type :input.")
    (assert (= 1 (count outp)) "Please give exactly one layer the type :output.")
    (let [nn (FreeformNetwork.)
          hm (loop [hm {} lrs layers]  ;; We do not know the id of the object we create
               (if (empty? lrs)  ;;TODO: Maybe it can be done much more easily
                 hm
                 (let [l (first lrs)
                       key (:id l)
                       neurons (:neurons l)
                       conn (condp = (:type l)
                              :input (. nn (createInputLayer neurons))
                              :hidden (. nn (createLayer neurons))
                              :output (. nn (createOutputLayer neurons)))]
                   (recur (assoc hm key conn) (rest lrs)))))]
      ;; Create the connections
      (doseq [conn connections]
        (let [l1-object (hm (:id (:layer1 conn)))
              l2-object (hm (:id (:layer2 conn)))
              activ-fn (:activation-function (:layer2 conn))
              bias (:bias conn)
              rec (:recurrent? conn)]
          (. nn (connectLayers l1-object l2-object activ-fn bias rec))))
      (. nn reset)
      ;;Return the network
      nn)))
