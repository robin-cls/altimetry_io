from cnes_alti_reader import CnesAltiCatalog


def test_catalog_init():
    catalog = CnesAltiCatalog.load()

    assert catalog

    # TODO: Replace by real tests
    catalog.show_dataset(dtypes="cls_table")
    catalog.show_dataset(dtypes="sc_collection")
    catalog.show_dataset(dtypes="file_collection")
    catalog.show_dataset(dtypes=["cls_table", "file_collection"])

    catalog.show_dataset(dtypes=["cls_table", "file_collection"], containing="jason")

    catalog.show_dataset(containing="jason")
    catalog.show_dataset(containing="something_not_existing")
