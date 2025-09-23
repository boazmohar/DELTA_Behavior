using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Gimbl;

public class LogService : MonoBehaviour
{
    // Services.
    private LoggerObject logger;

    // Cue Log Message format.
    public class CueStateMsg            // Message for logging current cue state.
    {
        public int id;
        public int id2;
        public bool isRewarding;
        public int position;
    }
    public CueStateMsg stateMsg = new CueStateMsg();

    // Cue Result Message.
    public class CueResultMsg           // Message for when cue undergoes reset to log the result of the traversal.
    {
        public int id;
        public int id2;
        public int position;
        public bool isRewarding;
        public bool hasGivenReward;
        public int numLicksInReward;
        public int numLicksInPre;
    }
    public CueResultMsg resultMsg = new CueResultMsg();

    // Period Message.
    public class PeriodMsg
    {
        public string periodType;
        public float duration;              
        public string cueSet;               
        public bool isGuided;               
    }
    public PeriodMsg periodMsg = new PeriodMsg();

    // Start is called before the first frame update
    void Start()
    {
        // Get instance of logger.
        logger = FindObjectOfType<LoggerObject>();
    }

    public void LogLick() { logger.logFile.Log("Lick"); }
    public void LogReward() { logger.logFile.Log("Reward"); }

    // Log cue states.
    /// <summary>
    /// Logs the position, rewarding condition, and id of newly initialized cues.
    /// </summary>
    /// <param name="position">Z position of cue</param>
    /// <param name="isRewarding">rewarding state</param>
    /// <param name="id">cue id</param>
    /// <param name="id2">cue id2</param>
    public void LogState(Cue cue)
    {
        stateMsg.position = (int)(cue.transform.position.z * 1000);
        stateMsg.isRewarding = cue.isRewarding;
        stateMsg.id = cue.id;
        stateMsg.id2 = cue.id2;
        logger.logFile.Log("Cue State", stateMsg);
    }

    // Log Result
    /// <summary>
    /// Log result when cue undergoes reset to log the result of the traversal.
    /// </summary>
    /// <param name="position">z position of the cue</param>
    /// <param name="isRewarding">reward state</param>
    /// <param name="id">cue id</param>
    /// <param name="id2">cue id2</param>
    /// <param name="hasGivenReward">if reward was dispensed</param>
    /// <param name="numLicksInPre">Number of licks in Pre trigger zone</param>
    /// <param name="numLicksInReward">Number of licks in reward zone</param>
    public void LogResult(Cue cue)
    {
        resultMsg.position = (int)(cue.transform.position.z * 1000);
        resultMsg.id = cue.id;
        resultMsg.id2 = cue.id2;
        resultMsg.isRewarding = cue.isRewarding;
        resultMsg.hasGivenReward = cue.hasGivenReward;
        resultMsg.numLicksInPre = cue.numLicksInPre;
        resultMsg.numLicksInReward = cue.numLicksInReward;
        logger.logFile.Log("Cue Result", resultMsg);
    }

    /// <summary>
    /// Used to store period parameters with message (e.g. "Start Period" or "End Period")
    /// </summary>
    /// <param name="period"> period to log info from</param>
    /// <param name="msg">message to log with the informatioh (e.g. "Start Period")</param>
    public void LogPeriodMsg(SessionSettings.SessionPeriod period, string msg)
    {
        // Set message.
        periodMsg.periodType = period.periodType.ToString();
        periodMsg.duration = period.duration;
        if (period.periodType==SessionSettings.SessionPeriod.TaskPeriodType.TASK)
        {
            periodMsg.cueSet = period.cueSet.name;
            periodMsg.isGuided = period.isGuided;
        }
        else
        {
            periodMsg.cueSet = null;
            periodMsg.isGuided = false;
        }

        logger.logFile.Log(msg, periodMsg);
    }
    public void BallLock(bool status)
    {
        if (status) { logger.logFile.Log("Ball Lock On"); }
        else { logger.logFile.Log("Ball Lock Off"); }
    }
}
