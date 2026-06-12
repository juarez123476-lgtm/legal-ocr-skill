---
name: legal-ocr
description: >
  Extrai e analisa texto de documentos jurídicos brasileiros escaneados em PDF.
  Usa PaddleOCR + EasyOCR com dicionário jurídico especializado (CC, CLT, LGPD,
  Lei de Franáquias). Identifica estrutura (relatório, fundamentação, dispositivo),
  valida qualidade e sinaliza páginas que precisam de revisão humana.

  ACIONAR quando o usuário precisar extrair texto de um PDF escaneado de processo
  judicial, sentença, acórdão, petição, contrato ou qualquer documento jurídico
  digitalizado via scanner.

  NÃO acionar para PDFs nativos digitais (use ferramentas de extração direta).
allowed-tools:
  - Bash
  - Read
  - Glob
---

# Legal OCR — Extração de Documentos Jurídicos

Quando esta skill for acionada, execute os passos abaixo na ordem indicada.
O argumento principal é o caminho do PDF. Argumentos opcionais são:

| Argumento | Padrão | Descrição |
|---|---|---|
| `--quality standard\|high` | `standard` | `high` para documentos antigos ou deteriorados |
| `--dpi N` | `300` | DPI da conversão (use `400` para baixa qualidade) |
| `--no-gpu` | GPU ativada | Desabilita aceleração por GPU |
| `--confidence-threshold N` | `0.3` | Confiança mínima para aceitar resultado |

---

## Passo 1 — Validar o ambiente

Verifique se `pipeline_ocr.py` e as dependências principais estão disponíveis:

```bash
python3 -c "import paddleocr, cv2, fitz; print('OK')" 2>&1
```

Se retornar erro, execute primeiro:

```bash
bash setup.sh
```

---

## Passo 2 — Validar o arquivo de entrada

```bash
python3 -c "
import fitz, sys
try:
    doc = fitz.open('$PDF_PATH')
    print(f'Páginas: {len(doc)}')
    doc.close()
except Exception as e:
    print(f'ERRO: {e}'); sys.exit(1)
"
```

Se o arquivo não existir ou não for um PDF válido, informar ao usuário e parar.

---

## Passo 3 — Executar o OCR

```bash
python3 pipeline_ocr.py "$PDF_PATH" \
  --output "${PDF_PATH%.pdf}_ocr.json" \
  $EXTRA_ARGS
```

Mostrar progresso ao usuário enquanto processa.

---

## Passo 4 — Apresentar o resumo

Leia o JSON gerado e apresente o seguinte resumo:

```
📄 Resultado OCR — [nome do arquivo]

Páginas processadas : N
Qualidade geral      : excellent / good / acceptable / poor
Confiança média      : XX%
Páginas p/ revisão   : N
Tempo               : Xs
Engine              : PaddleOCR (fallback EasyOCR em N páginas)

Estrutura identificada:
  [x] Cabeçalho      — [primeiros 80 chars]
  [x] Relatório
  [x] Fundamentação
  [x] Dispositivo
  [ ] Assinaturas

Problemas detectados:
  - [lista de issues e warnings, se houver]

Texto extraído (primeiros 500 chars):
  ...

Arquivo JSON completo: [caminho]
```

Se `requires_review` for `true` em alguma página, listar quais páginas precisam
de revisão humana.

---

## Tratamento de erros

| Situação | Ação |
|---|---|
| Arquivo não encontrado | Informar e pedir caminho correto |
| PDF nativo (não escaneado) | Avisar que PDF nativo não precisa de OCR |
| GPU sem memória | Sugerir re-executar com `--no-gpu` |
| Confiança geral < 50% | Sugerir `--quality high --dpi 400` |
| Dependências ausentes | Executar `bash setup.sh` |

---

## Tipos de documento suportados

| Tipo | Legislação | Estrutura esperada |
|---|---|---|
| Sentença / Acórdão | CPC | Relatório → Fundamentação → Dispositivo |
| Contrato civil | CC Lei 10.406/2002 | Cláusulas, partes, objeto |
| Rescisão trabalhista | CLT art. 477 §6º | Verbas rescisórias, assinaturas |
| Acordo de dados | LGPD Lei 13.709/2018 | Controlador, operador, titular |
| COF / Franquia | Lei 13.966/2019 | Prazo mínimo 14 dias |
| Licença de software | Lei 9.609/1998 art. 3 | Objeto, royalties, prazo |
| Licença de patente | LPI Lei 9.279/1996 | Cessionário, royalties |
| Locatício | Lei do Inquilinato 8.245/1991 | Partes, imóvel, aluguel |

---

## Exemplos de invocação

```
/legal-ocr sentenca_escaneada.pdf
/legal-ocr acordao_antigo.pdf --quality high --dpi 400
/legal-ocr contrato_lgpd.pdf --confidence-threshold 0.2
/legal-ocr cof_franquia.pdf --no-gpu
```
