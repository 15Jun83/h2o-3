package water.api;


import water.jdbc.SQLManager;

/**
 * Import Sql Table into H2OFrame
 */

public class ImportSQLTableHandler extends Handler {

  @SuppressWarnings("unused") // called through reflection by RequestServer
  public ImportSQLTableV99 importSQLTable(int version, ImportSQLTableV99 importSqlTable) {
    String[] key = new String[1];
    SQLManager.importSqlTable(importSqlTable.JDBCDriver, importSqlTable.host, importSqlTable.port, 
            importSqlTable.database, importSqlTable.table, importSqlTable.username, importSqlTable.password, key);
    
    importSqlTable.destination_frame = key[0];
    return importSqlTable;
  }

}
