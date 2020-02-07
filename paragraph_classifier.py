import logging
import numpy as np
import pickle

def load_classifier_model(model_path='../Paragraph_classification/paragraph_classifer_model'):
    """
    load model from pickled model file
    """
    with open(model_path, 'rb') as f:
        svc, decision_tree, random_forest, mlp, vectorizer = pickle.load(f)
    return [svc, decision_tree, random_forest, mlp, vectorizer]


def transform_user_features(paragraph):
    """
    feature extract from paragraph by human design
    
    paragraph - str, paragraph
    
    feature - numpy array, features of paragraph
    """
    
    numbers = sum(c.isdigit() for c in paragraph)

    features = [
      bool(numbers >= 1),
      bool(numbers > 5),
      bool(numbers > 10),

      #Heading heuristics
      bool('experiment' in paragraph[:50].lower()),
      bool('synthesi' in paragraph[:50].lower()),
      bool('prepar' in paragraph[:50].lower()),
      bool('abstract' in paragraph[:50].lower()),
      bool('characteriz' in paragraph[:50].lower()),

      #Domain keyword checks
      bool('heat' in paragraph.lower()),
      bool('zr' in paragraph.lower()),
      bool('hf' in paragraph.lower()),
      bool('cool' in paragraph.lower()),
      bool('dissolve' in paragraph.lower()),
      bool('mix' in paragraph.lower()),
      bool('obtain' in paragraph.lower()),
      bool('stir' in paragraph.lower()),
      bool('dropwise' in paragraph.lower()),
      bool('formed' in paragraph.lower()),
      bool('wash' in paragraph.lower()),
      bool('sealed' in paragraph.lower()),
      bool('yield' in paragraph.lower()),
         

      bool('mol ' in paragraph.lower()),
      bool('%' in paragraph.lower()),
      bool('ml ' in paragraph.lower()),
      bool(' ph ' in paragraph.lower()),

      bool('ratio' in paragraph.lower()),
      bool('stoichiometric' in paragraph.lower()),

      bool('sample' in paragraph.lower()),
      bool('solution' in paragraph.lower()),
      bool('product' in paragraph.lower()),
      bool('chemical' in paragraph.lower()),

      bool('study' in paragraph.lower()),
      bool('method' in paragraph.lower()),
      bool('technique' in paragraph.lower()),
      bool('route' in paragraph.lower()),
      bool('procedure' in paragraph.lower()),

      #Heuristic phrases
      bool('was prepared by' in paragraph.lower()),
      bool('dissolved in' in paragraph.lower()),
      bool('final product' in paragraph.lower()),
      bool('the precursors' in paragraph.lower()),
      bool('purchased from' in paragraph.lower()),

      #Characterization
      bool('xrd' in paragraph.lower()),
      bool('ftir' in paragraph.lower()),
      bool('voltammetry' in paragraph.lower()),
      bool('sem ' in paragraph.lower()),
      bool('microscop' in paragraph.lower()),
      bool('spectroscop' in paragraph.lower()),
    ]
    #return np.array([])
    return np.array(features)


def predict_if_synthese(paragraph, model=None, thresholds=2):
    """
    Predict if a paragraph is about synthese.

    We use 4 different classifier: svm, decision tree, random forest and multilayer perceptron.

    The result of prediction is yes if the sum of predict result is higher than $thresholds.
    
    Param:
        paragraph - str, raw content of paragraph
        model - list, 4 different model load by load_classifier_model(). 
                if this argv is None, then load it automaticly. 
                Note: load these model every time will cost much time, so if you are going
                      to do a batch job, it is better to load model in advance.

    Returns:
        result - result for prediction, 1 for yes, 0 for no
    """
    if model == None:
        model = load_classifier_model()

    svc, decision_tree, random_forest, mlp, vectorizer = model

    cv_vectors  = vectorizer.transform([paragraph]).toarray()
    ud_vectors = np.array(transform_user_features(paragraph)).reshape(1, -1)
    features = np.hstack([ud_vectors, cv_vectors])
    
    result = [svc.predict(features)[0], 
#               decision_tree.predict(features)[0], 
              random_forest.predict(features)[0], 
              mlp.predict(features)[0]]

    if sum(result) <= thresholds:
        logging.debug('\n{}\nclassifer output: {}, thresholds: {}, is synthese paragraph: Yes'.format(paragraph, ' '.join([str(i) for i in result]), thresholds))
        return 1
    else:
        logging.debug('\n{}\nclassifer output: {}, thresholds: {}, is synthese paragraph: No'.format(paragraph, ' '.join([str(i) for i in result]), thresholds))
        return 0


