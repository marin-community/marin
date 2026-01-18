import dspy
from experiments.dspy import metrics_fhir

MATCH_THRESHOLD = 0.7

def claim_verification_metric(example: dspy.Example, prediction: dspy.Prediction, trace = None) -> float:
    return int(example.label == prediction.label_int)


def field_extraction_metric(example: dspy.Example, prediction: dspy.Prediction, trace = None) -> float:
    """
    Evaluate field extraction by comparing prediction with gold standard.
    Returns a score between 0.0 and 1.0 based on field-level accuracy.
    """
    try:
        # Get record_id from example
        record_id = getattr(example, "record_id", None)
        if record_id is None:
            return 0.0
        
        # Load gold standard data
        gold_data = metrics_fhir.load_gold_data()
        if record_id not in gold_data:
            return 0.0
        
        gold_record = gold_data[record_id]
        
        # Get result from prediction
        result_record = getattr(prediction, "result", None)
        if result_record is None:
            # Try to get it from prediction dict if it's a dict-like object
            if hasattr(prediction, "toDict"):
                result_record = prediction.toDict().get("result")
            elif isinstance(prediction, dict):
                result_record = prediction.get("result")
        
        if result_record is None:
            return 0.0
        
        # Evaluate all field categories
        total_matches = 0
        total_fields = 0
        
        patient_matches, patient_total = metrics_fhir.evaluate_patient_fields(gold_record, result_record)
        total_matches += patient_matches
        total_fields += patient_total
        
        practitioner_matches, practitioner_total = metrics_fhir.evaluate_practitioner_fields(gold_record, result_record)
        total_matches += practitioner_matches
        total_fields += practitioner_total
        
        immunization_matches, immunization_total = metrics_fhir.evaluate_immunization_fields(gold_record, result_record)
        total_matches += immunization_matches
        total_fields += immunization_total
        
        allergy_matches, allergy_total = metrics_fhir.evaluate_allergy_fields(gold_record, result_record)
        total_matches += allergy_matches
        total_fields += allergy_total
        
        # Return accuracy as a float between 0.0 and 1.0
        if total_fields == 0:
            return 0.0
        
        return (total_matches / total_fields) > MATCH_THRESHOLD
    
    except Exception as e:
        # Return 0.0 on any error
        return 0.0