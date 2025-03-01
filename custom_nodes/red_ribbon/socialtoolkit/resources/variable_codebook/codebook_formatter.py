"""
Codebook formatter for VariableCodebook
"""
import logging
from typing import Dict, Any, Optional, List
import json
from .codebook_models import Codebook, CodebookGroup, Variable, CategoryValue, DataType

logger = logging.getLogger(__name__)

class CodebookFormatter:
    """
    Formats codebook data for output in different formats
    """
    
    def __init__(self):
        logger.info("CodebookFormatter initialized")
    
    def format(self, codebook: Codebook, format_type: str = "json") -> Optional[str]:
        """
        Format a codebook in the specified format
        
        Args:
            codebook: Codebook object to format
            format_type: Type of formatting to apply (json, html, markdown, csv)
            
        Returns:
            Formatted codebook string or None if formatting failed
        """
        logger.info(f"Formatting codebook as {format_type}")
        
        try:
            if format_type == "json":
                return self._format_as_json(codebook)
                
            elif format_type == "html":
                return self._format_as_html(codebook)
                
            elif format_type == "markdown":
                return self._format_as_markdown(codebook)
                
            elif format_type == "csv":
                return self._format_as_csv(codebook)
                
            else:
                logger.error(f"Unknown format type: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error formatting codebook: {str(e)}")
            return None
    
    def _format_as_json(self, codebook: Codebook) -> str:
        """Format codebook as JSON"""
        
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                d = obj.__dict__.copy()
                
                # Handle special cases
                if isinstance(obj, Codebook):
                    d["created_at"] = d["created_at"].isoformat()
                    d["updated_at"] = d["updated_at"].isoformat()
                
                elif isinstance(obj, CodebookGroup):
                    d["created_at"] = d["created_at"].isoformat()
                    d["updated_at"] = d["updated_at"].isoformat()
                
                elif isinstance(obj, Variable):
                    d["created_at"] = d["created_at"].isoformat()
                    d["updated_at"] = d["updated_at"].isoformat()
                    if isinstance(d["data_type"], DataType):
                        d["data_type"] = d["data_type"].value
                        
                return d
            return str(obj)
        
        return json.dumps(codebook, default=serialize, indent=2)
    
    def _format_as_html(self, codebook: Codebook) -> str:
        """Format codebook as HTML"""
        html = f"""
        <div class="codebook">
            <h1>{codebook.name} (v{codebook.version})</h1>
            <p>{codebook.description or ''}</p>
            
            <h2>Variables</h2>
        """
        
        # Group variables by their group
        for group in codebook.groups:
            html += f"""
            <div class="codebook-group">
                <h3>{group.name}</h3>
                <p>{group.description or ''}</p>
                <table class="variables-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Description</th>
                            <th>Type</th>
                            <th>Required</th>
                            <th>Values</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for var in group.variables:
                # Handle category values for categorical variables
                values = ""
                if var.categories:
                    values = "<ul>"
                    for cat in var.categories:
                        values += f"<li><code>{cat.code}</code>: {cat.label}</li>"
                    values += "</ul>"
                
                html += f"""
                    <tr>
                        <td><code>{var.name}</code></td>
                        <td>{var.description or ''}</td>
                        <td>{var.data_type.value}</td>
                        <td>{'Yes' if var.required else 'No'}</td>
                        <td>{values}</td>
                    </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        # Include any variables not in groups
        ungrouped = [v for v in codebook.variables if not any(v in g.variables for g in codebook.groups)]
        if ungrouped:
            html += """
            <div class="codebook-group">
                <h3>Ungrouped Variables</h3>
                <table class="variables-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Description</th>
                            <th>Type</th>
                            <th>Required</th>
                            <th>Values</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for var in ungrouped:
                # Handle category values for categorical variables
                values = ""
                if var.categories:
                    values = "<ul>"
                    for cat in var.categories:
                        values += f"<li><code>{cat.code}</code>: {cat.label}</li>"
                    values += "</ul>"
                
                html += f"""
                    <tr>
                        <td><code>{var.name}</code></td>
                        <td>{var.description or ''}</td>
                        <td>{var.data_type.value}</td>
                        <td>{'Yes' if var.required else 'No'}</td>
                        <td>{values}</td>
                    </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _format_as_markdown(self, codebook: Codebook) -> str:
        """Format codebook as Markdown"""
        md = f"# {codebook.name} (v{codebook.version})\n\n"
        
        if codebook.description:
            md += f"{codebook.description}\n\n"
        
        # Group variables by their group
        for group in codebook.groups:
            md += f"## {group.name}\n\n"
            
            if group.description:
                md += f"{group.description}\n\n"
            
            md += "| Name | Description | Type | Required | Values |\n"
            md += "|------|-------------|------|----------|--------|\n"
            
            for var in group.variables:
                # Handle category values for categorical variables
                values = ""
                if var.categories:
                    values = ", ".join([f"`{cat.code}`: {cat.label}" for cat in var.categories])
                
                required = "Yes" if var.required else "No"
                md += f"| `{var.name}` | {var.description or ''} | {var.data_type.value} | {required} | {values} |\n"
            
            md += "\n"
        
        # Include any variables not in groups
        ungrouped = [v for v in codebook.variables if not any(v in g.variables for g in codebook.groups)]
        if ungrouped:
            md += "## Ungrouped Variables\n\n"
            
            md += "| Name | Description | Type | Required | Values |\n"
            md += "|------|-------------|------|----------|--------|\n"
            
            for var in ungrouped:
                # Handle category values for categorical variables
                values = ""
                if var.categories:
                    values = ", ".join([f"`{cat.code}`: {cat.label}" for cat in var.categories])
                
                required = "Yes" if var.required else "No"
                md += f"| `{var.name}` | {var.description or ''} | {var.data_type.value} | {required} | {values} |\n"
        
        return md
    
    def _format_as_csv(self, codebook: Codebook) -> str:
        """Format codebook variables as CSV"""
        lines = ["group,name,description,data_type,required,min_value,max_value,categories"]
        
        # Add all variables with their group
        for group in codebook.groups:
            for var in group.variables:
                categories = ""
                if var.categories:
                    categories = ";".join([f"{cat.code}={cat.label}" for cat in var.categories])
                
                required = "1" if var.required else "0"
                min_val = str(var.min_value) if var.min_value is not None else ""
                max_val = str(var.max_value) if var.max_value is not None else ""
                
                line = f'"{group.name}","{var.name}","{var.description or ""}","{var.data_type.value}",{required},{min_val},{max_val},"{categories}"'
                lines.append(line)
        
        # Include any variables not in groups
        ungrouped = [v for v in codebook.variables if not any(v in g.variables for g in codebook.groups)]
        for var in ungrouped:
            categories = ""
            if var.categories:
                categories = ";".join([f"{cat.code}={cat.label}" for cat in var.categories])
            
            required = "1" if var.required else "0"
            min_val = str(var.min_value) if var.min_value is not None else ""
            max_val = str(var.max_value) if var.max_value is not None else ""
            
            line = f'"","{var.name}","{var.description or ""}","{var.data_type.value}",{required},{min_val},{max_val},"{categories}"'
            lines.append(line)
        
        return "\n".join(lines)