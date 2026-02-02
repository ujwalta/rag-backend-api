"""
Document processing utilities for extracting text from various file types.
"""
from typing import Optional
import io
from pathlib import Path
import PyPDF2
import pdfplumber


class DocumentProcessor:
    """Handles extraction of text from different document types."""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes, use_pdfplumber: bool = True) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_content: PDF file content as bytes
            use_pdfplumber: Use pdfplumber (better for tables) vs PyPDF2
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If text extraction fails
        """
        try:
            if use_pdfplumber:
                return DocumentProcessor._extract_with_pdfplumber(file_content)
            else:
                return DocumentProcessor._extract_with_pypdf2(file_content)
        except Exception as e:
            # Fallback to alternative method
            try:
                if use_pdfplumber:
                    return DocumentProcessor._extract_with_pypdf2(file_content)
                else:
                    return DocumentProcessor._extract_with_pdfplumber(file_content)
            except Exception as fallback_error:
                raise Exception(
                    f"Failed to extract PDF text: {str(e)}. "
                    f"Fallback also failed: {str(fallback_error)}"
                )
    
    @staticmethod
    def _extract_with_pdfplumber(file_content: bytes) -> str:
        """Extract text using pdfplumber (better for structured documents)."""
        text_parts = []
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_with_pypdf2(file_content: bytes) -> str:
        """Extract text using PyPDF2 (faster, simpler)."""
        text_parts = []
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes, encoding: str = 'utf-8') -> str:
        """
        Extract text from TXT file.
        
        Args:
            file_content: TXT file content as bytes
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Extracted text content
            
        Raises:
            UnicodeDecodeError: If encoding is incorrect
        """
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            # Try common alternative encodings
            for alt_encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return file_content.decode(alt_encoding)
                except UnicodeDecodeError:
                    continue
            
            # Last resort: decode with error handling
            return file_content.decode('utf-8', errors='replace')
    
    @staticmethod
    def extract_text(
        file_content: bytes, 
        file_type: str,
        **kwargs
    ) -> str:
        """
        Extract text from file based on type.
        
        Args:
            file_content: File content as bytes
            file_type: File extension (e.g., '.pdf', '.txt')
            **kwargs: Additional arguments for specific extractors
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            Exception: If extraction fails
        """
        file_type = file_type.lower()
        
        if file_type == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(file_content, **kwargs)
        elif file_type == '.txt':
            return DocumentProcessor.extract_text_from_txt(file_content, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def validate_file(
        filename: str, 
        file_size: int, 
        max_size_mb: int = 10,
        allowed_extensions: set = {'.pdf', '.txt'}
    ) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            max_size_mb: Maximum allowed size in MB
            allowed_extensions: Set of allowed file extensions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"File type {file_ext} not allowed. Allowed: {allowed_extensions}"
        
        # Check size
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"File size {file_size} exceeds maximum {max_size_mb}MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, None