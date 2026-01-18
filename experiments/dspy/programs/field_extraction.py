import dspy

from typing import Literal
from pydantic import BaseModel, Field

class PatientNote(BaseModel):
    record_id: int
    note: str

class PersonNameAndTitle(BaseModel):
    family: str | None = Field(default=None, description="surname")
    given: list[str] | None = Field(default=None, description="Given names (first and middle names)")
    prefix: str | None = Field(default=None, description="title of the person, e.g., Dr., Mr., Mrs., Ms.")

class Address(BaseModel):
    line: str | None = Field(default=None, alias="street")
    city: str | None
    state: str | None
    postalCode: str | None
    country: Literal["US"] | None

class Practitioner(BaseModel):
    name: PersonNameAndTitle | None
    phone: str | None = Field(default=None, description="Phone number of the healthcare provider")
    email: str | None = Field(default=None, description="Email address of the healthcare provider")
    address: Address | None = Field(default=None, description="Address of the healthcare provider")

class Immunization(BaseModel):
    traits: list[str] | None = Field(default=None, description="Text describing the name and traits of the immunization")
    status: Literal["completed"] | None = Field(default=None, description="If no traits are present, then the status cannot be determined")
    occurrenceDate: str | None = Field(default=None, description="ISO-8601 format for date")

class Substance(BaseModel):
    category: Literal["environment", "food", "medication", "other"]
    name: str | None
    manifestation: str | None = Field(default=None, description="Text describing the manifestation of the allergy or intolerance")

class Allergy(BaseModel):
    substance: list[Substance] = Field(description="Substances the patient is allergic to")

class Patient(BaseModel):
    record_id: int | None = Field(default=None)
    name: PersonNameAndTitle | None
    age: int | None
    gender: Literal["male", "female"] | None
    birthDate: str | None = Field(default=None, description="ISO-8601 format for date")
    phone: str | None = Field(default=None, description="Phone number of the patient")
    email: str | None = Field(default=None, description="Email address of the patient")
    maritalStatus: Literal["Married", "Divorced", "Widowed", "NeverMarried"] | None
    address: Address | None = Field(default=None, description="Residence address of the patient")
    allergy: list[Allergy] | None = Field(default=None)

class PatientInfo(dspy.Signature):
    """
    - Do not infer any information that is not explicitly mentioned in the text.
    - If you are unsure about any field, leave it as None.
    """
    note: str = dspy.InputField()
    patient: Patient = dspy.OutputField()

class PractitionerInfo(dspy.Signature):
    """
    - Do not infer any information that is not explicitly mentioned in the text.
    - If you are unsure about any field, leave it as None.
    """
    note: str = dspy.InputField()
    practitioner: list[Practitioner] | None = dspy.OutputField()

class ImmunizationInfo(dspy.Signature):
    """
    Extracts immunization information from a patient note.
    """
    note: str = dspy.InputField(desc="Immunization info only")
    immunization: list[Immunization] | None = dspy.OutputField()

class FieldExtraction(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use TypedPredictor if available, otherwise standard Predict (which might need handling for pydantic)
        # For this implementation we'll assume dspy.TypedPredictor or similar support exists or we use Predict
        # and rely on dspy to handle Pydantic output fields which it does in recent versions.
        # If TypedPredictor is preferred:
        self.extract_patient = dspy.Predict(PatientInfo)
        self.extract_practitioner = dspy.Predict(PractitionerInfo)
        self.extract_immunization = dspy.Predict(ImmunizationInfo)

    def forward(self, note: str, record_id: int):
        # Note: Original code handled async, we'll do sync for simplicity in tracing unless async is needed
        p_pred = self.extract_patient(note=note)
        pr_pred = self.extract_practitioner(note=note)
        i_pred = self.extract_immunization(note=note)

        # Helper to extract model dump safely
        patient_data = p_pred.patient.model_dump() if p_pred.patient else {}
        patient_data['record_id'] = record_id

        practitioner_data = [item.model_dump() for item in pr_pred.practitioner] if pr_pred.practitioner else []
        immunization_data = [item.model_dump() for item in i_pred.immunization] if i_pred.immunization else []

        result = {
            "patient": patient_data,
            "practitioner": practitioner_data,
            "immunization": immunization_data
        }
        return dspy.Prediction(result=result)
