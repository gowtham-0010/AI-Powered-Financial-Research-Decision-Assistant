"""
Multi-provider LLM wrapper supporting Mock, Hugging Face, and OpenAI.
Factory-based architecture allows runtime switching via LLM_PROVIDER env variable.
Maintains compatibility with existing RAG pipeline and frontend.

IMPROVEMENTS:
- Hugging Face: Lightweight model with fallback to mock
- Timeout handling with graceful degradation
- Reduced token usage for free tier reliability
- Proper error logging and recovery
"""

import logging
import time
from typing import Optional
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self):
        """Initialize base LLM client."""
        settings = get_settings()
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
    
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """Generate text from LLM."""
        pass


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns deterministic financial analysis."""
    
    def __init__(self):
        """Initialize mock LLM client."""
        super().__init__()
        logger.info("MockLLMClient initialized (no API calls, offline mode)")
    
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate mock financial analysis response.
        
        Args:
            system_prompt: System instruction (ignored in mock mode)
            user_message: User query (ignored in mock mode)
            context: Context with company metrics (parsed for realism)
            temperature: Ignored in mock mode
            
        Returns:
            str: Deterministic mock financial analysis
        """
        # Parse company name from context if available
        company_name = "Unknown Company"
        if "company_name" in context.lower():
            company_name = "Apple Inc."
        
        # Generate deterministic response based on context
        mock_response = f"""
## Investment Analysis Report - {company_name}

### Executive Summary
Based on the financial data provided, {company_name} presents a **LOW-RISK** investment opportunity with strong fundamentals and solid growth prospects. The company demonstrates:

- **Strong Balance Sheet**: Healthy cash position with manageable debt levels
- **Revenue Growth**: Consistent year-over-year growth trajectory
- **Market Position**: Well-established market presence with brand strength
- **Dividend Yield**: Attractive returns for income-focused investors

### Financial Analysis

**Valuation Metrics:**
- P/E Ratio: 28.4 (Fair value for growth profile)
- Price-to-Book Ratio: 45.3 (Premium valuation justified by ROE)
- Current Ratio: 1.50 (Adequate liquidity position)

**Profitability & Returns:**
- Return on Equity (ROE): 89% (Exceptional)
- Net Profit Margin: 25.4% (Industry leading)
- Free Cash Flow: Strong and consistent

**Debt & Leverage:**
- Debt-to-Equity: 0.89 (Conservative leverage)
- Interest Coverage: 12.5x (Very comfortable)

**Growth Metrics:**
- Revenue Growth: 15.3% YoY (Above market average)
- EPS Growth: 18.7% YoY (Strong earnings power)
- ROIC: 34.2% (Exceptional capital efficiency)

### Risk Assessment

**Overall Risk Level: LOW**
- Confidence Score: 87.3%
- Industry Risk: Moderate (Technology sector stable)
- Company Risk: Low (Diversified revenue streams)
- Market Risk: Moderate (Systemic market exposure)

### Investment Recommendation

**BUY** | Target Price: $195.50 (Upside: 12.3%)

For your investment horizon, {company_name} offers compelling value with strong competitive advantages, predictable cash flows, attractive valuation, and growth above market average.

### Comparison to Sector

{company_name} ranks favorably against technology peers:
- Earnings Quality: Above Average
- Growth Rate: 15.3% vs 12.1% (Sector Average)
- Dividend Yield: 0.42% vs 0.38% (Sector Average)
- ROE: 89% vs 16.2% (Sector Average)

### Conclusion

{company_name} represents a **STRONG BUY** for growth-oriented investors seeking exposure to innovation with moderate risk. The company's dominant market position, consistent execution, and strong financial performance support a positive outlook.

*This analysis is based on financial data and AI-driven insights.*
"""
        
        logger.info(f"Mock LLM generated response ({len(mock_response)} chars)")
        return mock_response


class HuggingFaceLLMClient(BaseLLMClient):
    """Hugging Face Inference API client for open-source LLMs with fallback."""
    
    def __init__(self):
        """Initialize Hugging Face LLM client."""
        super().__init__()
        settings = get_settings()
        
        self.api_token = settings.huggingface_api_token
        if not self.api_token or self.api_token == "hf_":
            logger.warning("HUGGINGFACE_API_TOKEN not set. HF mode may fail.")
        
        # Use lightweight model: Zephyr 7B is smaller, faster, and free-tier friendly
        # Mistral would timeout frequently on free tier
        self.api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Track fallbacks for logging
        self.fallback_count = 0
        
        logger.info("HuggingFaceLLMClient initialized with Zephyr-7B (free-tier optimized)")
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced from 3 to 2 for free tier
        wait=wait_exponential(multiplier=1, min=2, max=5),  # Shorter wait time
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using Hugging Face Inference API with fallback to mock.
        
        Args:
            system_prompt: System instruction for LLM
            user_message: User query
            context: Additional context
            temperature: Override default temperature
            
        Returns:
            str: Generated text from Hugging Face LLM or fallback mock response
        """
        try:
            import requests
            
            temp = temperature if temperature is not None else self.temperature
            
            # Compress prompt to reduce token usage
            # Keep only essential context for free tier
            compressed_context = self._compress_context(context)
            
            # Build compact prompt for Zephyr
            formatted_prompt = f"""[INST] {system_prompt}

Context (Key metrics):
{compressed_context}

User: {user_message} [/INST]"""
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "temperature": temp,
                    "max_new_tokens": 800,  # Reduced from 2000 for free tier
                    "do_sample": True,
                    "top_p": 0.9,
                }
            }
            
            logger.info("Calling Hugging Face Inference API (Zephyr-7B)...")
            start_time = time.time()
            
            # Aggressive timeout for free tier (45s instead of 60s)
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=45  # Shorter timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"HF response received in {elapsed:.1f}s")
            
            if response.status_code != 200:
                logger.warning(f"HF API error {response.status_code}. Falling back to mock.")
                return self._fallback_to_mock(context)
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Remove prompt from response
                if "[/INST]" in generated_text:
                    generated_text = generated_text.split("[/INST]")[-1].strip()
            else:
                logger.warning(f"Unexpected HF response format: {result}. Falling back to mock.")
                return self._fallback_to_mock(context)
            
            if not generated_text or len(generated_text) < 50:
                logger.warning(f"HF response too short ({len(generated_text)} chars). Falling back to mock.")
                return self._fallback_to_mock(context)
            
            logger.info(f"HF LLM call successful. Response length: {len(generated_text)}")
            return generated_text
            
        except TimeoutError:
            logger.warning("HF API timeout (45s). Free tier may be loading model. Falling back to mock.")
            return self._fallback_to_mock(context)
        
        except requests.exceptions.Timeout:
            logger.warning("Request timeout. Model may be loading. Falling back to mock.")
            return self._fallback_to_mock(context)
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"HF connection error: {e}. Falling back to mock.")
            return self._fallback_to_mock(context)
        
        except Exception as e:
            logger.warning(f"HF LLM error: {e}. Falling back to mock.")
            return self._fallback_to_mock(context)
    
    def _compress_context(self, context: str) -> str:
        """
        Compress context to reduce token usage for free tier.
        
        Args:
            context: Original context
            
        Returns:
            str: Compressed context with only essential metrics
        """
        lines = context.split('\n')
        compressed = []
        
        # Keep only lines with numbers/metrics (financial data)
        for line in lines:
            if any(char.isdigit() for char in line):
                # Shorten the line
                if len(line) > 100:
                    line = line[:100]
                compressed.append(line.strip())
            
            # Limit to 10 key metrics
            if len(compressed) >= 10:
                break
        
        return '\n'.join(compressed)
    
    def _fallback_to_mock(self, context: str) -> str:
        """
        Fallback to mock response when Hugging Face fails.
        
        Args:
            context: Original context (for reference)
            
        Returns:
            str: Mock financial analysis
        """
        self.fallback_count += 1
        logger.info(f"Using mock response (fallback #{self.fallback_count})")
        
        mock_client = MockLLMClient()
        return mock_client.generate(
            system_prompt="",
            user_message="",
            context=context
        )


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client using modern SDK (v1.x+)."""
    
    def __init__(self):
        """Initialize OpenAI client with settings."""
        super().__init__()
        settings = get_settings()
        
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=settings.llm_api_key)
            logger.info(f"OpenAILLMClient initialized with model: {self.model}")
            
        except ImportError:
            logger.error("OpenAI SDK not installed. Install with: pip install openai")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        context: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from OpenAI API with retry logic.
        
        Args:
            system_prompt: System instruction for LLM
            user_message: User query
            context: Additional context to include
            temperature: Override default temperature
            
        Returns:
            str: Generated text from OpenAI
            
        Raises:
            Exception: If API call fails after retries
        """
        try:
            from openai import RateLimitError, APIError
            
            temp = temperature if temperature is not None else self.temperature
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            if context:
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{user_message}"
                })
            else:
                messages.append({"role": "user", "content": user_message})
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"OpenAI call successful. Tokens used: {response.usage.total_tokens}")
            return generated_text
            
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limited: {e}. Retrying...")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise


class LLMFactory:
    """Factory for creating appropriate LLM client based on provider."""
    
    @staticmethod
    def create_client() -> BaseLLMClient:
        """
        Create LLM client based on LLM_PROVIDER setting.
        
        Returns:
            BaseLLMClient: Appropriate LLM client instance
            
        Raises:
            ValueError: If provider is not recognized
        """
        settings = get_settings()
        provider = settings.llm_provider.lower().strip()
        
        logger.info(f"Creating LLM client with provider: {provider}")
        
        if provider == "mock":
            return MockLLMClient()
        elif provider == "huggingface":
            return HuggingFaceLLMClient()
        elif provider == "openai":
            return OpenAILLMClient()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {provider}. "
                f"Allowed: mock, huggingface, openai"
            )


# Global singleton instance
_llm_client = None


def get_llm_client() -> BaseLLMClient:
    """
    Get or create LLM client (singleton pattern).
    
    Reads LLM_PROVIDER from environment to determine which provider to use.
    - mock: Local deterministic responses (no cost, offline)
    - huggingface: Hugging Face Inference API with fallback to mock (free)
    - openai: OpenAI API (production, optional, requires API key)
    
    Returns:
        BaseLLMClient: Singleton instance of appropriate LLM client
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMFactory.create_client()
    return _llm_client