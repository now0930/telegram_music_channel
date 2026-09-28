import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from music_intent import vocabulary, validate_intent, fallback_intent, rank_candidates, choose_intent


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.collection = Mock()
        self.collection.get.return_value = {'metadatas': [
            {'mood': '신남', 'genre_fixed': '댄스', 'era': '2020년대',
             'bpm_range': '업비트 (Allegro)', 'is_instrumental': 'False'},
            {'mood': '잔잔함', 'genre_fixed': '', 'era': '미상', 'is_instrumental': 'True'}]}
        self.values = vocabulary(self.collection)

    def test_schema_uses_existing_values_and_string_boolean(self):
        client = Mock()
        intent = fallback_intent('가을', '오전')
        intent.update(mood='신남', is_instrumental='False')
        import json
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=json.dumps(intent)))
        self.assertEqual(choose_intent(client, 'mica', {}, '오전', self.values), intent)
        schema = client.chat.call_args.kwargs['format']['properties']
        self.assertEqual(schema['is_instrumental']['enum'], [None, 'False', 'True'])
        self.assertNotIn('미상', schema['era']['enum'])

    def test_invalid_model_values_rejected(self):
        for field, value in [('mood', '신나는'), ('is_instrumental', True), ('genre_fixed', 'invented')]:
            intent = fallback_intent('가을', '오전')
            intent[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_intent(intent, self.values)

    def test_ranking_prefers_metadata_preserves_vector_ties(self):
        intent = fallback_intent('가을', '오전')
        intent.update(mood='신남', genre_fixed='댄스')
        candidates = [('a', {'mood': '잔잔함'}), ('b', {'mood': '신남'}),
                      ('c', {'mood': '신남', 'genre_fixed': '댄스'}), ('d', {'genre_fixed': '댄스'})]
        self.assertEqual([p for p, _ in rank_candidates(candidates, intent)], ['c', 'b', 'd', 'a'])
        self.assertEqual(rank_candidates(candidates, fallback_intent('가을', '오전')), candidates)

    def test_reason_not_empty_and_extra_fields_rejected(self):
        intent = fallback_intent('가을', '오전')
        intent['count'] = 100
        with self.assertRaises(ValueError):
            validate_intent(intent, self.values)


if __name__ == '__main__':
    unittest.main()
