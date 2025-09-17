# core/medicines/bd.py
"""Bangladesh medicine information and pricing"""

import logging
from typing import Dict, Any
from utils.web_search import WebSearchTool

logger = logging.getLogger(__name__)

class BangladeshMedicineInfo:
    """Handles Bangladesh-specific medicine information and pricing"""
    
    def __init__(self):
        self.web = WebSearchTool()
    
    def get_medicine_info(self, query: str, include_price: bool = False) -> str:
        """Get comprehensive medicine information for Bangladesh"""
        try:
            # Resolve Bangladesh medicine details
            medicine_data = self.web.resolve_bd_medicine(query)
            
            if medicine_data.get("status") != "success":
                return self._fallback_response(query)
            
            # Build response
            response_parts = []
            
            # Basic information
            brand = medicine_data.get("brand", "")
            generic = medicine_data.get("generic", "")
            form = medicine_data.get("form", "")
            strength = medicine_data.get("strength", "")
            company = medicine_data.get("company", "")
            
            if brand or generic:
                info_line = "**Medicine Information (Bangladesh):**\n"
                if brand:
                    info_line += f"• Brand: {brand}\n"
                if generic:
                    info_line += f"• Generic: {generic}\n"
                if form:
                    info_line += f"• Form: {form}\n"
                if strength:
                    info_line += f"• Strength: {strength}\n"
                if company:
                    info_line += f"• Manufacturer: {company}\n"
                
                response_parts.append(info_line)
            
            # Pricing information
            if include_price and brand:
                price_info = self.web.get_bd_medicine_price(brand)
                if price_info.get("status") == "success" and price_info.get("price"):
                    price_line = f"\n**Price in Bangladesh:** {price_info['price']}"
                    response_parts.append(price_line)
            
            # General medical information
            search_terms = [query]
            if generic and generic != query:
                search_terms.append(generic)
            
            medical_info = self._get_general_medicine_info(" ".join(search_terms))
            if medical_info:
                response_parts.append(f"\n**Medical Information:**\n{medical_info}")
            
            # Source attribution
            source = medicine_data.get("source", "medex.com.bd")
            response_parts.append(f"\n*Source: {source}*")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Bangladesh medicine info error: {e}")
            return self._fallback_response(query)
    
    def _get_general_medicine_info(self, medicine_name: str) -> str:
        """Get general medical information about the medicine"""
        try:
            # This would typically use RAG or knowledge base
            # For now, return a generic structure
            return (
                f"• **Uses:** Consult medical literature or healthcare provider\n"
                f"• **Dosage:** Follow prescription or package instructions\n"
                f"• **Safety:** Check for allergies and drug interactions\n"
                f"• **Storage:** Store as directed on package"
            )
        except Exception:
            return ""
    
    def _fallback_response(self, query: str) -> str:
        """Fallback response when medicine lookup fails"""
        return (
            f"I couldn't find specific Bangladesh information for '{query}'. "
            "This might be a less common medicine or the name might need adjustment. "
            "Please check the spelling or try using the generic name. "
            "For accurate information, consult a pharmacist or healthcare provider."
        )
    
    def get_formulary_card(self, query: str) -> str:
        """Get formulary-style medicine card"""
        try:
            medicine_data = self.web.resolve_bd_medicine(query)
            
            if medicine_data.get("status") != "success":
                return self._fallback_response(query)
            
            # Create formulary card format
            card_parts = []
            
            # Header
            brand = medicine_data.get("brand", "")
            generic = medicine_data.get("generic", "")
            
            if brand and generic:
                card_parts.append(f"**{brand}** ({generic})")
            elif brand:
                card_parts.append(f"**{brand}**")
            elif generic:
                card_parts.append(f"**{generic}**")
            
            # Details
            details = []
            if medicine_data.get("form"):
                details.append(f"Form: {medicine_data['form']}")
            if medicine_data.get("strength"):
                details.append(f"Strength: {medicine_data['strength']}")
            if medicine_data.get("company"):
                details.append(f"Manufacturer: {medicine_data['company']}")
            
            if details:
                card_parts.append(" | ".join(details))
            
            # Pricing
            if brand:
                price_info = self.web.get_bd_medicine_price(brand)
                if price_info.get("status") == "success" and price_info.get("price"):
                    card_parts.append(f"**Price:** {price_info['price']}")
            
            # Safety note
            card_parts.append("\n*Always consult healthcare providers for medical advice.*")
            
            return "\n".join(card_parts)
            
        except Exception as e:
            logger.error(f"Formulary card error: {e}")
            return self._fallback_response(query)
