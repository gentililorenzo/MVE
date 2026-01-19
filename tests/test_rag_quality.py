import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock import per dimostrazione (sostituisci con il tuo vero import)
try:
    sys.path.append('..')
    from src.prompt_hybrid import HybridPromptSystem
except ImportError:
    logger.warning("Modulo src.prompt_hybrid non trovato. Assicurati del path.")

class RAGEvaluator:
    def __init__(self, test_file: str = 'test_cases.json'):
        self.system = HybridPromptSystem()
        self.test_file = test_file
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    def load_test_cases(self) -> List[Dict]:
        """Carica i test case da file esterno JSON o usa quelli di default."""
        try:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File {self.test_file} non trovato. Uso casi di default.")
            return [
                {
                    'id': 1,
                    'company': {'dimensione': 15, 'attivita': 'Falegnameria artigianale'},
                    'question': 'Come ridurre i consumi energetici?',
                    'expected_keywords': ['energia', 'macchinari', 'mwh'] # Usa lowercase qui
                }
                # Aggiungi altri casi qui...
            ]

    def evaluate_response(self, response: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """
        Valuta la risposta. 
        TODO: Sostituire con Semantic Similarity (es. BERT score) per produzione.
        """
        response_lower = response.lower()
        found = [kw for kw in expected_keywords if kw.lower() in response_lower]
        score = len(found) / len(expected_keywords) if expected_keywords else 0
        
        return {
            'score': score,
            'found': found,
            'missing': list(set(expected_keywords) - set(found))
        }

    def run_tests(self):
        test_cases = self.load_test_cases()
        report = []
        
        print(f"\n🧪 AVVIO VALUTAZIONE SU {len(test_cases)} CASI\n" + "="*60)

        for test in test_cases:
            t_id = test.get('id', 'N/A')
            print(f"Test ID: {t_id} | Azienda: {test['company']['attivita']}")
            
            try:
                start = time.time()
                response = self.system.consult(test['company'], test['question'])
                elapsed = time.time() - start
                
                eval_metrics = self.evaluate_response(response, test['expected_keywords'])
                
                result_entry = {
                    'test_id': t_id,
                    'status': 'SUCCESS',
                    'question': test['question'],
                    'response_snippet': response[:100] + "...",
                    'time_sec': round(elapsed, 2),
                    **eval_metrics
                }
                
                # Feedback visivo immediato
                icon = "✅" if eval_metrics['score'] > 0.7 else "⚠️" if eval_metrics['score'] > 0.4 else "❌"
                print(f"{icon} Score: {eval_metrics['score']:.2%} ({elapsed:.2f}s)")
                
            except Exception as e:
                logger.error(f"Errore nel test {t_id}: {str(e)}")
                result_entry = {
                    'test_id': t_id,
                    'status': 'ERROR',
                    'error': str(e),
                    'score': 0
                }
                print(f"❌ ERRORE: {str(e)}")

            report.append(result_entry)
            print("-" * 60)

        self.save_report(report)

    def save_report(self, results: List[Dict]):
        """Salva i risultati in un file JSON timestampato."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"rag_eval_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'total_tests': len(results),
            'avg_score': sum(r.get('score', 0) for r in results) / len(results) if results else 0,
            'avg_time': sum(r.get('time_sec', 0) for r in results) / len(results) if results else 0,
            'details': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"\n📊 Report salvato in: {filename}")
        print(f"Score Medio Totale: {summary['avg_score']:.2%}")

if __name__ == "__main__":
    # Esempio di utilizzo
    evaluator = RAGEvaluator()
    evaluator.run_tests()