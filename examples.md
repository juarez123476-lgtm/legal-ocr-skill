# Legal OCR - Exemplos de Uso

Este documento contém exemplos práticos de como usar a skill Legal OCR em diferentes cenários.

## Índice

1. [Uso Básico](#uso-básico)
2. [Processamento em Lote](#processamento-em-lote)
3. [Integração com SIGEDEC](#integração-com-sigedec)
4. [Uso Programático (Python)](#uso-programático-python)
5. [Casos Especiais](#casos-especiais)
6. [Troubleshooting](#troubleshooting)

---

## Uso Básico

### Exemplo 1: Processar um único PDF

```bash
# Uso mais simples
python .claude/skills/legal-ocr/pipeline_ocr.py sentenca_123.pdf

# Output: sentenca_123_ocr.json
```

**Resultado esperado:**
- Arquivo JSON com texto extraído
- Confiança média: ~90%
- Tempo: ~3-5 segundos por página

### Exemplo 2: Especificar arquivo de saída

```bash
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input acórdão_stj_2024.pdf \
  --output resultado_acordao.json
```

### Exemplo 3: Aumentar qualidade para documentos antigos

```bash
# Use --quality high para documentos de baixa qualidade
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input sentenca_antiga_1995.pdf \
  --quality high \
  --dpi 400
```

**Quando usar `--quality high`:**
- Documentos escaneados antes de 2000
- PDFs com manchas ou desgaste
- Documentos com baixo contraste
- Aumenta processamento em ~30% mas melhora acurácia em 15-20%

### Exemplo 4: Processar sem GPU

```bash
# Se não tiver GPU ou quiser usar apenas CPU
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input documento.pdf \
  --no-gpu
```

---

## Processamento em Lote

### Exemplo 5: Processar múltiplos PDFs de um diretório

```python
#!/usr/bin/env python3
"""
Batch OCR processor for multiple PDFs
"""

import os
import sys
from pathlib import Path
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

def batch_process(input_dir, output_dir):
    """Process all PDFs in a directory"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Initialize pipeline once (reuse for all documents)
    pipeline = LegalDocumentOCRPipeline(
        use_gpu=True,
        quality='standard'
    )

    # Find all PDFs
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")

    # Process each
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        output_path = Path(output_dir) / f"{pdf_path.stem}_ocr.json"

        try:
            result = pipeline.process_legal_document(
                str(pdf_path),
                output_json_path=str(output_path)
            )

            print(f"  ✓ Success: {result['quality_summary']['overall_quality']}")
            print(f"  ✓ Confidence: {result['quality_summary']['avg_confidence']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

if __name__ == '__main__':
    batch_process(
        input_dir='./processos_escaneados/',
        output_dir='./textos_extraidos/'
    )
```

**Uso:**
```bash
python batch_processor.py
```

### Exemplo 6: Processamento paralelo com multiprocessing

```python
#!/usr/bin/env python3
"""
Parallel batch OCR with multiprocessing
"""

import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

def process_single_pdf(args):
    """Process a single PDF (worker function)"""
    pdf_path, output_dir, use_gpu = args

    # Each process gets its own pipeline instance
    pipeline = LegalDocumentOCRPipeline(
        use_gpu=use_gpu,
        quality='standard'
    )

    output_path = Path(output_dir) / f"{Path(pdf_path).stem}_ocr.json"

    try:
        result = pipeline.process_legal_document(
            pdf_path,
            output_json_path=str(output_path)
        )
        return {'success': True, 'file': pdf_path, 'result': result}
    except Exception as e:
        return {'success': False, 'file': pdf_path, 'error': str(e)}

def parallel_batch_process(input_dir, output_dir, num_workers=None):
    """Process multiple PDFs in parallel"""

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Find all PDFs
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files")

    # Determine number of workers
    if num_workers is None:
        # Use CPU count - 1 (leave one core free)
        num_workers = max(1, cpu_count() - 1)

    print(f"Using {num_workers} parallel workers")

    # Prepare args for each PDF
    # Note: GPU usage might be limited if multiple processes share same GPU
    args_list = [(str(pdf), output_dir, False) for pdf in pdf_files]

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_pdf, args_list)

    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}: {r['error']}")

if __name__ == '__main__':
    parallel_batch_process(
        input_dir='./processos_escaneados/',
        output_dir='./textos_extraidos/',
        num_workers=4
    )
```

---

## Integração com SIGEDEC

### Exemplo 7: Processar e indexar no Qdrant

```python
#!/usr/bin/env python3
"""
Process OCR and index in Qdrant vector database
"""

import sys
import json
from pathlib import Path
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import openai

# Initialize
ocr_pipeline = LegalDocumentOCRPipeline(use_gpu=True)
qdrant_client = QdrantClient(url="http://localhost:6333")
openai.api_key = "your-api-key-here"

def get_embedding(text):
    """Get text embedding from OpenAI"""
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response['data'][0]['embedding']

def process_and_index(pdf_path):
    """Process PDF and index in Qdrant"""

    print(f"Processing: {pdf_path}")

    # 1. Extract text with OCR
    result = ocr_pipeline.process_legal_document(pdf_path)

    # 2. Check quality
    if result['quality_summary']['overall_quality'] == 'poor':
        print("  ⚠ Poor quality - skipping indexing")
        return

    # 3. Get full text
    full_text = result['full_text']

    # 4. Extract metadata
    metadata = {
        'filename': result['filename'],
        'num_pages': result['metadata']['total_pages'],
        'confidence': result['quality_summary']['avg_confidence'],
        'extracted_at': result['timestamp'],
        'source': 'ocr'
    }

    # 5. Generate embedding
    print("  Generating embedding...")
    embedding = get_embedding(full_text[:8000])  # Limit to 8k chars

    # 6. Index in Qdrant
    collection_name = "jurisprudencia_brasileira"

    # Create collection if not exists
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    # Upload point
    point = PointStruct(
        id=hash(pdf_path),  # Simple ID based on filename
        vector=embedding,
        payload={
            'text': full_text,
            'metadata': metadata,
            'structure': result.get('document_structure', {})
        }
    )

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point]
    )

    print(f"  ✓ Indexed successfully")

if __name__ == '__main__':
    pdf_path = 'acordao_stj_2024.pdf'
    process_and_index(pdf_path)
```

### Exemplo 8: Integração com workflow n8n

```javascript
// n8n Node: Execute OCR Python Script
{
  "name": "Legal OCR",
  "type": "n8n-nodes-base.executeCommand",
  "parameters": {
    "command": "python",
    "arguments": [
      ".claude/skills/legal-ocr/pipeline_ocr.py",
      "={{$json.pdf_path}}",
      "--output", "={{$json.output_path}}",
      "--quality", "high"
    ]
  }
}

// n8n Node: Process OCR Result
{
  "name": "Parse OCR JSON",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": `
      const fs = require('fs');
      const result = JSON.parse(fs.readFileSync($json.output_path, 'utf-8'));

      // Check quality
      if (result.quality_summary.avg_confidence < 0.7) {
        throw new Error('OCR quality too low - manual review required');
      }

      // Extract key fields
      return {
        case_number: extractCaseNumber(result.full_text),
        parties: extractParties(result.document_structure),
        decision: result.document_structure.dispositivo,
        full_text: result.full_text,
        metadata: result.metadata
      };
    `
  }
}
```

---

## Uso Programático (Python)

### Exemplo 9: Usar a classe LegalDocumentOCRPipeline diretamente

```python
#!/usr/bin/env python3
"""
Direct usage of OCR pipeline in Python
"""

import sys
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

# Initialize pipeline
pipeline = LegalDocumentOCRPipeline(
    use_gpu=True,
    quality='high',
    confidence_threshold=0.4,
    use_fallback=True
)

# Process document
result = pipeline.process_legal_document(
    pdf_path='sentenca_civil.pdf',
    dpi=300,
    output_json_path='resultado.json'
)

# Access results
print(f"Total pages: {result['metadata']['total_pages']}")
print(f"Average confidence: {result['quality_summary']['avg_confidence']}")
print(f"Overall quality: {result['quality_summary']['overall_quality']}")

# Get full text
full_text = result['full_text']
print(f"\nExtracted text ({len(full_text)} characters):")
print(full_text[:500])

# Check document structure
structure = result['document_structure']
if structure['dispositivo']:
    print(f"\nDispositivo encontrado:")
    print(structure['dispositivo'][:300])
```

### Exemplo 10: Processar apenas páginas específicas

```python
#!/usr/bin/env python3
"""
Process only specific pages of a PDF
"""

import sys
import fitz
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

def process_pages(pdf_path, page_numbers):
    """Process only specified pages"""

    pipeline = LegalDocumentOCRPipeline(use_gpu=True)

    # Extract specific pages to temporary PDF
    pdf = fitz.open(pdf_path)
    temp_pdf = fitz.open()

    for page_num in page_numbers:
        if 0 <= page_num < len(pdf):
            temp_pdf.insert_pdf(pdf, from_page=page_num, to_page=page_num)

    temp_path = 'temp_pages.pdf'
    temp_pdf.save(temp_path)
    temp_pdf.close()
    pdf.close()

    # Process
    result = pipeline.process_legal_document(temp_path)

    # Cleanup
    import os
    os.remove(temp_path)

    return result

# Example: Process only pages 5-10
result = process_pages('documento_grande.pdf', page_numbers=[5, 6, 7, 8, 9, 10])
print(f"Processed {len(result['pages'])} pages")
```

---

## Casos Especiais

### Exemplo 11: Documentos multi-coluna

```python
#!/usr/bin/env python3
"""
Handle multi-column documents (e.g., Diário Oficial)
"""

import sys
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

# Use standard quality but increase DPI for better column separation
pipeline = LegalDocumentOCRPipeline(
    use_gpu=True,
    quality='standard'
)

result = pipeline.process_legal_document(
    pdf_path='diario_oficial_2024.pdf',
    dpi=400  # Higher DPI helps with column detection
)

# PaddleOCR automatically handles column detection
print(f"Processed {result['metadata']['total_pages']} pages")
```

### Exemplo 12: Documentos com tabelas

```python
#!/usr/bin/env python3
"""
Extract tables from legal documents
"""

import sys
import json
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline
from paddleocr import PPStructure

def extract_with_tables(pdf_path):
    """Extract text and tables"""

    # Standard text extraction
    pipeline = LegalDocumentOCRPipeline(use_gpu=True)
    text_result = pipeline.process_legal_document(pdf_path)

    # Table extraction with PaddleOCR Structure
    table_engine = PPStructure(
        table=True,
        lang='pt',
        show_log=False
    )

    images = pipeline.pdf_to_images(pdf_path)

    tables_extracted = []
    for i, img in enumerate(images):
        result = table_engine(img)

        for item in result:
            if item['type'] == 'table':
                tables_extracted.append({
                    'page': i + 1,
                    'html': item['res']['html'],
                    'bbox': item['bbox']
                })

    return {
        'text': text_result,
        'tables': tables_extracted
    }

result = extract_with_tables('processo_com_tabelas.pdf')
print(f"Found {len(result['tables'])} tables")
```

### Exemplo 13: Documentos muito antigos (baixíssima qualidade)

```python
#!/usr/bin/env python3
"""
Maximum quality settings for very old documents
"""

import sys
sys.path.append('.claude/skills/legal-ocr')

from pipeline_ocr import LegalDocumentOCRPipeline

# Custom pipeline with aggressive preprocessing
pipeline = LegalDocumentOCRPipeline(
    use_gpu=True,
    quality='high',            # High quality preprocessing
    confidence_threshold=0.2,  # Lower threshold (accept more results)
    use_fallback=True          # Always try fallback
)

result = pipeline.process_legal_document(
    pdf_path='sentenca_1950.pdf',
    dpi=600,  # Very high DPI for maximum detail
    output_json_path='resultado_antigo.json'
)

# Flag pages that need manual review
pages_to_review = [
    p['page_num'] for p in result['pages']
    if p.get('validation', {}).get('requires_review', False)
]

print(f"Pages requiring manual review: {pages_to_review}")
```

---

## Troubleshooting

### Problema 1: GPU out of memory

```python
# Solution 1: Reduce batch size (edit pipeline_ocr.py)
# Solution 2: Use CPU instead
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input documento.pdf \
  --no-gpu
```

### Problema 2: Baixa confiança em todas as páginas

```bash
# Try increasing DPI and using high quality mode
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input documento.pdf \
  --quality high \
  --dpi 400 \
  --confidence-threshold 0.2
```

### Problema 3: Erros de dependências

```bash
# Reinstall all dependencies
cd .claude/skills/legal-ocr
pip uninstall -y paddleocr easyocr
pip install -r requirements.txt --upgrade

# Download Portuguese models again
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='pt')"
```

### Problema 4: Texto extraído está embaralhado (ordem errada)

```python
# Multi-column or complex layout issue
# Increase DPI to improve layout detection
python .claude/skills/legal-ocr/pipeline_ocr.py \
  --input documento.pdf \
  --dpi 400
```

---

## Resumo de Comandos Úteis

```bash
# Uso padrão
python pipeline_ocr.py documento.pdf

# Alta qualidade
python pipeline_ocr.py documento.pdf --quality high --dpi 400

# Sem GPU
python pipeline_ocr.py documento.pdf --no-gpu

# Threshold baixo (aceitar mais resultados)
python pipeline_ocr.py documento.pdf --confidence-threshold 0.2

# Sem fallback (apenas PaddleOCR)
python pipeline_ocr.py documento.pdf --no-fallback

# Output customizado
python pipeline_ocr.py documento.pdf --output meu_resultado.json
```

---

**Para mais informações:**
- `SKILL.md` - Visão geral e conceitos
- `reference.md` - Detalhes técnicos completos
- `logs/ocr_processing.log` - Logs de processamento
