
#ifndef SMILEAPI_H
#define SMILEAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SMILE_SUCCESS,             SMILE_FAIL,                SMILE_INVALID_ARG,         SMILE_INVALID_STATE,       SMILE_COMP_NOT_FOUND,      SMILE_LICENSE_FAIL,        SMILE_CONFIG_PARSE_FAIL,   SMILE_CONFIG_INIT_FAIL,    SMILE_NOT_WRITTEN,       } smileres_t;

typedef enum {
  SMILE_UNINITIALIZED,       SMILE_INITIALIZED,         SMILE_RUNNING,             SMILE_ENDED              } smilestate_t;

typedef struct smileobj_t_ smileobj_t;

typedef enum {
  SMILE_LOG_MESSAGE = 1,
  SMILE_LOG_WARNING = 2,
  SMILE_LOG_ERROR = 3,
  SMILE_LOG_DEBUG = 4,
  SMILE_LOG_PRINT = 5
} smilelogtype_t;

typedef struct {
  smilelogtype_t type;       int level;                 const char *text;          const char *module;      } smilelogmsg_t;

typedef void (*LogCallback)(smileobj_t *smileobj, smilelogmsg_t message, void *param);

typedef void (*StateChangedCallback)(smileobj_t *smileobj, smilestate_t state, void *param);

typedef bool (*ExternalSinkCallback)(const float *data, long vectorSize, void *param);
struct sExternalSinkMetaDataEx;
typedef bool (*ExternalSinkCallbackEx)(const float *data, long nT, long N, const sExternalSinkMetaDataEx *metaData, void *param);

class cComponentMessage;
typedef bool (*ExternalMessageInterfaceCallback)(const cComponentMessage *msg, void *param);

typedef bool (*ExternalMessageInterfaceJsonCallback)(const char *msg, void *param);

typedef struct {
  const char *name;
  const char *value;
} smileopt_t;

smileobj_t *smile_new();

smileres_t smile_initialize(smileobj_t *smileobj, const char *configFile, int nOptions, const smileopt_t *options, int loglevel=2, int debug=0, int consoleOutput=0, const char *logFile=0);

smileres_t smile_run(smileobj_t *smileobj);

smileres_t smile_abort(smileobj_t *smileobj);

smileres_t smile_reset(smileobj_t *smileobj);

smileres_t smile_set_log_callback(smileobj_t *smileobj, LogCallback callback, void *param);

smilestate_t smile_get_state(smileobj_t *smileobj);

smileres_t smile_set_state_callback(smileobj_t *smileobj, StateChangedCallback callback, void *param);

void smile_free(smileobj_t *smileobj);

smileres_t smile_extsource_write_data(smileobj_t *smileobj, const char *componentName, const float *data, int nFrames);

smileres_t smile_extsource_set_external_eoi(smileobj_t *smileobj, const char *componentName);

smileres_t smile_extaudiosource_write_data(smileobj_t *smileobj, const char *componentName, const void *data, int length);

smileres_t smile_extaudiosource_set_external_eoi(smileobj_t *smileobj, const char *componentName);

smileres_t smile_extsink_set_data_callback(smileobj_t *smileobj, const char *componentName, ExternalSinkCallback callback, void *param);
smileres_t smile_extsink_set_data_callback_ex(smileobj_t *smileobj, const char *componentName, ExternalSinkCallbackEx callback, void *param);

smileres_t smile_extsink_get_num_elements(smileobj_t *smileobj, const char *componentName, long *numElements);
smileres_t smile_extsink_get_element_name(smileobj_t *smileobj, const char *componentName, long idx, const char **elementName);

smileres_t smile_extmsginterface_set_msg_callback(smileobj_t *smileobj, const char *componentName, ExternalMessageInterfaceCallback callback, void *param);

smileres_t smile_extmsginterface_set_json_msg_callback(smileobj_t *smileobj, const char *componentName, ExternalMessageInterfaceJsonCallback callback, void *param);

const char *smile_error_msg(smileobj_t *smileobj);

#ifdef __cplusplus
}
#endif

#endif
