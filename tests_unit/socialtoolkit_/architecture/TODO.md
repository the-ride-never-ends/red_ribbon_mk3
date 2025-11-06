# TODO: Fix Meta-Tester Violations in test_document_retrieval_from_websites.py

**Test File:** `/home/kylerose1946/red_ribbon_mk3/tests-unit/socialtoolkit_/architecture/test_document_retrieval_from_websites.py`

**Meta-Tester Results:** 247 failed tests, 1037 passed tests

---

## 1. Method Length Violations (35 tests)

**Issue:** Tests exceed 10 lines of code.

**Affected Tests:**
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_when_single_domain_processed_then_url_generator_called_once`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_single_domain_url_is_expanded_to_multiple_page_urls_1`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration_1`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration_1`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings_1`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings_2`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_empty_raw_data_results_in_empty_extraction`
- `TestDocumentsAreCreatedwithURLContext.test_documents_include_source_url`
- `TestDocumentsAreCreatedwithURLContext.test_documents_include_source_url_1`
- `TestDocumentsAreCreatedwithURLContext.test_multiple_strings_create_multiple_documents`
- `TestDocumentsAreCreatedwithURLContext.test_multiple_strings_create_multiple_documents_1`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_generator_creates_embeddings`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_generator_creates_embeddings_1`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_dimensions_match_configuration`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_1`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_2`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_3`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_1`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_2`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_3`
- `TestBatchProcessingConfigurationIsRespected.test_large_document_sets_are_processed_in_batches`
- `TestBatchProcessingConfigurationIsRespected.test_large_document_sets_are_processed_in_batches_1`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_1`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_2`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_1`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_2`
- `TestUserAgentConfigurationIsApplied.test_custom_user_agent_is_sent_in_http_requests`

**Action Required:**
- Refactor each test to be 10 lines or fewer (excluding docstring)
- Extract setup logic into fixtures
- Remove unnecessary comments
- Simplify assertion logic

---

## 2. Multiple Assertions Per Test (27 violations)

**Issue:** Tests contain more than one assertion statement.

**Affected Tests:**
- `TestExecuteMethodAcceptsListofDomainURLs.test_when_execute_called_with_single_domain_url_then_documents_retrieved`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_when_single_domain_processed_then_url_generator_called_once`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_single_domain_url_is_expanded_to_multiple_page_urls_1`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration_1`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration_1`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings_1`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings_2`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_empty_raw_data_results_in_empty_extraction`
- `TestDocumentsAreCreatedwithURLContext.test_documents_include_source_url`
- `TestDocumentsAreCreatedwithURLContext.test_documents_include_source_url_1`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_generator_creates_embeddings`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_generator_creates_embeddings_1`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_dimensions_match_configuration`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_1`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_2`
- `TestMetadataIsGeneratedforAllDocuments.test_metadata_includes_document_properties_3`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_1`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_2`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_3`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_2`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_1`

**Action Required:**
- Split tests with multiple assertions into separate test methods
- Each test should verify exactly one behavior
- Use descriptive test names that reflect the single assertion

---

## 3. Constructor Initialization in Tests (7 violations)

**Issue:** Tests instantiate classes directly instead of using mocks.

**Affected Tests:**
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_1`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_2`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_1`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_2`
- `TestUserAgentConfigurationIsApplied.test_custom_user_agent_is_sent_in_http_requests`

**Action Required:**
- Remove direct instantiation of `Timeout()` and `HTTPError()` exceptions
- Remove direct instantiation of `DocumentRetrievalFromWebsites` in `test_custom_user_agent_is_sent_in_http_requests`
- Use fixture-provided instances instead
- Mock exception raising behavior rather than constructing exceptions

---

## 4. Missing F-Strings in Assertions (35 violations)

**Issue:** Assertion messages don't use f-strings.

**All assertions currently use string concatenation with `+` operator instead of f-strings.**

**Action Required:**
- Replace all assertion messages with f-string format
- Example: `assert x == y, f"Expected {y}, but got {x}"`

---

## 5. Missing Dynamic Content in F-Strings (35 violations)

**Issue:** Assertion f-strings don't include dynamic values.

**Currently many assertions have static messages or don't properly interpolate variables.**

**Action Required:**
- Ensure all f-strings include actual runtime values
- Include expected vs actual values in assertion messages
- Example: `assert count == 5, f"Expected 5 documents, but got {count}"`

---

## 6. Multiple Production Calls (10 violations)

**Issue:** Tests call production methods more than once.

**Affected Tests:**
- `TestStaticandDynamicWebpagesAreParsedAppropriately.test_when_dynamic_webpage_processed_then_dynamic_parser_called`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_when_single_domain_processed_then_url_generator_called_once`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration_1`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration`
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_follow_links_configuration_1`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_raw_html_is_extracted_to_text_strings`
- `TestDataExtractionConvertsRawDatatoStructuredStrings.test_empty_raw_data_results_in_empty_extraction`
- `TestVectorsAreGeneratedforAllDocuments.test_vector_dimensions_match_configuration`
- `TestDocumentsVectorsandMetadataAreStored.test_all_data_is_persisted_to_storage_3`

**Action Required:**
- Each test should call `execute()` exactly once
- Store result in variable and verify behavior through assertions
- Avoid multiple calls to production code within single test

---

## 7. Fake Test Detected (1 violation)

**Issue:** Test uses mock/fake without proper verification.

**Affected Test:**
- `TestURLGenerationExpandsDomainURLstoPageURLs.test_url_generator_respects_max_depth_configuration_1`

**Action Required:**
- Review test to ensure mock behavior is properly verified
- Add appropriate mock assertions
- Ensure test validates actual behavior, not just mock setup

---

## 8. Magic Literals (28 violations)

**Issue:** Tests contain hardcoded numbers and strings instead of using constants.

**Affected Tests:**
- Multiple tests across all test classes use magic numbers like `5`, `10`, `100`, `1536`
- Tests use magic strings like `"/page1"`, `"/timeout"`, `"Single text content"`

**Action Required:**
- Replace all magic numbers with named constants from `mock_constants` fixture
- Replace all magic strings with named constants
- Use descriptive variable names for computed values
- Example: Replace `5` with `mock_constants["EXPECTED_PAGE_COUNT"]`

---

## 9. External Resources Usage (6 violations)

**Issue:** Tests use real external resources instead of mocks.

**Affected Tests:**
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_1`
- `TestExecuteHandlesHTTPRequestFailures.test_timeout_during_webpage_fetch_is_handled_2`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_1`
- `TestExecuteHandlesHTTPRequestFailures.test_404_not_found_error_is_handled_gracefully_2`

**Action Required:**
- Remove `from requests.exceptions import Timeout, HTTPError`
- Mock exception behavior instead of importing real exception classes
- Use `Mock(side_effect=Exception("message"))` pattern

---

## 10. Test Naming Convention Violations (36 violations)

**Issue:** Test names don't follow `test_when_x_then_y` convention.

**Affected Tests:**
- All tests with `_1`, `_2`, `_3` suffixes don't follow convention
- Examples:
  - `test_single_domain_url_is_expanded_to_multiple_page_urls_1`
  - `test_url_generator_respects_max_depth_configuration`
  - `test_raw_html_is_extracted_to_text_strings`
  - `test_documents_include_source_url`
  - `test_vector_generator_creates_embeddings`
  - etc.

**Action Required:**
- Rename all tests to follow `test_when_<condition>_then_<outcome>` pattern
- Remove numeric suffixes
- Make test names describe precondition and postcondition clearly
- Example: `test_raw_html_is_extracted_to_text_strings` → `test_when_raw_html_processed_then_text_strings_returned`

---

## 11. Missing Production Method in Class Docstrings (8 classes)

**Issue:** Test class docstrings don't mention the production method being tested.

**Affected Classes:**
- `TestDataExtractionConvertsRawDatatoStructuredStrings`
- `TestDocumentsAreCreatedwithURLContext`
- `TestVectorsAreGeneratedforAllDocuments`
- `TestMetadataIsGeneratedforAllDocuments`
- `TestDocumentsVectorsandMetadataAreStored`
- `TestBatchProcessingConfigurationIsRespected`
- `TestExecuteHandlesHTTPRequestFailures`
- `TestUserAgentConfigurationIsApplied`

**Action Required:**
- Add "Tests for: ClassName.method_name()" to each class docstring
- Example: `"""Rule: X\n    Tests for: DocumentRetrievalFromWebsites.execute()\n    """`

---

## 12. Duplicate Assertions (1 violation)

**Issue:** File contains duplicate assertion logic across multiple tests.

**Affected File:** `test_document_retrieval_from_websites.py`

**Action Required:**
- Review all assertion patterns for duplicates
- Extract common assertion logic into helper methods
- Use parametrized tests where appropriate to reduce duplication
