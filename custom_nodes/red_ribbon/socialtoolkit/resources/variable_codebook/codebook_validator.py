"""
Codebook validator for VariableCodebook
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from .codebook_models import Codebook, Variable, DataType

logger = logging.getLogger(__name__)

class CodebookValidator:
    """
    Validates codebook data and variable definitions
    """
    
    def __init__(self):
        logger.info("CodebookValidator initialized")
    
    def validate_codebook(self, codebook: Codebook) -> Tuple[bool, List[str]]:
        """
        Validate a complete codebook
        
        Args:
            codebook: Codebook to validate
            
        Returns:
            Tuple of (is_valid, list_of_validation_messages)
        """
        logger.info(f"Validating codebook: {codebook.name}")
        
        is_valid = True
        messages = []
        
        # Basic codebook validation
        if not codebook.name:
            is_valid = False
            messages.append("Codebook name is required")
        
        if not codebook.version:
            is_valid = False
            messages.append("Codebook version is required")
            
        # Validate each variable
        for variable in codebook.variables:
            var_valid, var_messages = self.validate_variable(variable)
            if not var_valid:
                is_valid = False
                messages.extend([f"Variable '{variable.name}': {msg}" for msg in var_messages])
        
        # Check for duplicate variable names
        var_names = [v.name for v in codebook.variables]
        duplicates = [name for name in set(var_names) if var_names.count(name) > 1]
        if duplicates:
            is_valid = False
            messages.append(f"Duplicate variable names found: {', '.join(duplicates)}")
        
        # Check group references are valid
        for group in codebook.groups:
            if group.parent_group_id and group.parent_group_id not in [g.id for g in codebook.groups]:
                is_valid = False
                messages.append(f"Group '{group.name}' references non-existent parent group: {group.parent_group_id}")
        
        logger.info(f"Codebook validation result: {'valid' if is_valid else 'invalid'}")
        return is_valid, messages
    
    def validate_variable(self, variable: Variable) -> Tuple[bool, List[str]]:
        """
        Validate a variable definition
        
        Args:
            variable: Variable to validate
            
        Returns:
            Tuple of (is_valid, list_of_validation_messages)
        """
        is_valid = True
        messages = []
        
        # Basic validation
        if not variable.name:
            is_valid = False
            messages.append("Variable name is required")
        
        if not variable.data_type:
            is_valid = False
            messages.append("Data type is required")
            
        # Type-specific validation
        if variable.data_type == DataType.INTEGER or variable.data_type == DataType.FLOAT:
            # Check min/max consistency
            if variable.min_value is not None and variable.max_value is not None:
                if variable.min_value > variable.max_value:
                    is_valid = False
                    messages.append(f"Min value {variable.min_value} is greater than max value {variable.max_value}")
        
        # Categorical validation
        if variable.data_type == DataType.CATEGORICAL:
            if not variable.categories:
                is_valid = False
                messages.append("Categories are required for categorical variables")
            else:
                # Check for duplicate category codes
                cat_codes = [c.code for c in variable.categories]
                duplicate_codes = [code for code in set(cat_codes) if cat_codes.count(code) > 1]
                if duplicate_codes:
                    is_valid = False
                    messages.append(f"Duplicate category codes found: {', '.join(map(str, duplicate_codes))}")
                
                # Check for duplicate category labels
                cat_labels = [c.label for c in variable.categories]
                duplicate_labels = [label for label in set(cat_labels) if cat_labels.count(label) > 1]
                if duplicate_labels:
                    is_valid = False
                    messages.append(f"Duplicate category labels found: {', '.join(duplicate_labels)}")
        
        return is_valid, messages
