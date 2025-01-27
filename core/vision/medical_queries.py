"""
MedicalQueries: Specialized medical queries for surgical video analysis.
Provides detailed prompts for various aspects of colorectal surgery procedures.
"""

from typing import Dict, List

class SurgicalQueries:
    """Collection of specialized surgical queries for vision analysis."""
    
    @staticmethod
    def get_anatomical_queries() -> List[Dict[str, str]]:
        """Get queries for anatomical analysis."""
        return [
            {
                "id": "anatomy_identification",
                "query": "Identify and describe the anatomical structures visible "
                        "in this surgical field, including layers of tissue and "
                        "important landmarks."
            },
            {
                "id": "vascular_supply",
                "query": "Describe the vascular anatomy visible in this image, "
                        "including major vessels and their branches."
            },
            {
                "id": "tissue_planes",
                "query": "Identify the surgical planes and tissue layers being "
                        "dissected in this image."
            }
        ]

    @staticmethod
    def get_technical_queries() -> List[Dict[str, str]]:
        """Get queries for technical analysis."""
        return [
            {
                "id": "instrument_technique",
                "query": "Analyze the surgical technique being demonstrated, "
                        "including instrument positioning and tissue handling."
            },
            {
                "id": "dissection_technique",
                "query": "Describe the dissection technique being used, including "
                        "the direction and method of tissue separation."
            },
            {
                "id": "hemostasis_technique",
                "query": "Identify methods of hemostasis being employed and "
                        "describe their application."
            }
        ]

    @staticmethod
    def get_safety_queries() -> List[Dict[str, str]]:
        """Get queries for safety analysis."""
        return [
            {
                "id": "critical_structures",
                "query": "Identify critical structures at risk in this surgical "
                        "field and describe their protection measures."
            },
            {
                "id": "safety_zones",
                "query": "Describe the surgical safety zones and anatomical "
                        "boundaries being observed."
            },
            {
                "id": "complications_risk",
                "query": "Identify potential complications risks in this surgical "
                        "step and their prevention measures."
            }
        ]

    @staticmethod
    def get_pathology_queries() -> List[Dict[str, str]]:
        """Get queries for pathological analysis."""
        return [
            {
                "id": "tissue_pathology",
                "query": "Describe any visible pathological changes in the "
                        "tissue, including inflammation or neoplastic changes."
            },
            {
                "id": "margin_assessment",
                "query": "Assess the surgical margins and describe their "
                        "adequacy based on visible tissue characteristics."
            },
            {
                "id": "lymph_nodes",
                "query": "Identify and describe any visible lymph nodes and "
                        "their characteristics."
            }
        ]

    @staticmethod
    def get_procedure_specific_queries() -> Dict[str, List[Dict[str, str]]]:
        """Get procedure-specific queries."""
        return {
            "right_colectomy": [
                {
                    "id": "ileocolic_vessels",
                    "query": "Describe the approach to the ileocolic vessels "
                            "and their division technique."
                },
                {
                    "id": "right_mobilization",
                    "query": "Analyze the right colon mobilization technique "
                            "and plane of dissection."
                }
            ],
            "low_anterior_resection": [
                {
                    "id": "tme_plane",
                    "query": "Assess the total mesorectal excision plane and "
                            "quality of dissection."
                },
                {
                    "id": "pelvic_autonomics",
                    "query": "Identify and describe the preservation of pelvic "
                            "autonomic nerves."
                }
            ],
            "anastomosis": [
                {
                    "id": "anastomotic_technique",
                    "query": "Analyze the anastomotic technique, including "
                            "tissue approximation and stapler application."
                },
                {
                    "id": "blood_supply",
                    "query": "Assess the blood supply to the anastomotic segments."
                }
            ]
        }

    @staticmethod
    def get_quality_assessment_queries() -> List[Dict[str, str]]:
        """Get queries for surgical quality assessment."""
        return [
            {
                "id": "tissue_handling",
                "query": "Evaluate the quality of tissue handling and assess "
                        "for any trauma or inappropriate manipulation."
            },
            {
                "id": "plane_quality",
                "query": "Assess the quality of surgical plane development "
                        "and maintenance throughout the dissection."
            },
            {
                "id": "technical_precision",
                "query": "Evaluate the technical precision of the current "
                        "surgical step and adherence to oncologic principles."
            }
        ]
