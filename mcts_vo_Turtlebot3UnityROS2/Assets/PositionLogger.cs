using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Ground-truth position logging for the robot and every obstacle, gated by
// MCTSVO_POS_LOG_PATH inherited from the launching Python process
// (mctsVoRos/loopHandler_copy.py, --log-positions). Absent/empty -> Enabled
// is false and every call site's guard skips the call entirely.
//
// Binary, append-only, self-describing format - see mctsVoRos/position_log.py
// for the reader and format docs. The Unity player is always killed with
// SIGTERM/SIGKILL to its whole process group (loopHandler_copy.py never
// calls process.wait()), so there is no clean-shutdown opportunity and the
// file must tolerate truncation. Flushing is batched (every FlushEvery
// writes) rather than per-row: a synchronous flush is a real disk I/O stall,
// and GroundTruthOdometry logs on every physics tick with no accumulator -
// flushing that often measurably slowed Unity's frame time and, since the
// Python-side control loop is itself wall-clock-timed (it already holds
// position and repeats the last action on slow/stale sensor reads), changed
// experiment outcomes. Batching bounds data loss on an abrupt kill to at
// most FlushEvery-1 trailing records instead of eliminating it, in exchange
// for not perturbing the timing-sensitive run it's observing.
public static class PositionLogger
{
    private const string EnvVarName = "MCTSVO_POS_LOG_PATH";
    private static readonly byte[] Magic = { (byte)'M', (byte)'V', (byte)'P', (byte)'L' };
    private const byte FormatVersion = 2;
    private const int FlushEvery = 50;

    public static readonly bool Enabled;

    private static readonly Dictionary<string, ushort> _objectIds = new Dictionary<string, ushort>();
    private static ushort _nextObjectId = 0;
    private static BinaryWriter _writer;
    private static bool _openFailed = false;
    private static int _writesSinceFlush = 0;

    static PositionLogger()
    {
        string path = Environment.GetEnvironmentVariable(EnvVarName);
        Enabled = !string.IsNullOrEmpty(path);
        if (Enabled)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
                _writer = new BinaryWriter(stream);
                _writer.Write(Magic);
                _writer.Write(FormatVersion);
                _writer.Flush();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"PositionLogger: failed to open '{path}': {e.Message}");
                _writer = null;
                _openFailed = true;
            }
        }
    }

    // radius is the object's true world-space radius (e.g. transform.localScale.x / 2
    // for a unit sphere primitive), used by the Python side to draw real-size
    // circles instead of dots so collisions are visible. Pass 0f for the robot -
    // mctsVoRos/position_log.py uses the planner's own fixed robot_radius (0.15,
    // MCTS_VO/environment_creator.py) for that instead, matching what every
    // existing debug animation already draws the robot's collision circle as.
    public static void LogRow(string objectName, int step, float time,
                               float x, float z, float speedInstant, float speedMax, float radius)
    {
        if (!Enabled || _openFailed)
            return;

        ushort id = GetOrDefineObjectId(objectName);

        _writer.Write((byte)1);       // tag: data row
        _writer.Write(step);
        _writer.Write(time);
        _writer.Write(id);
        _writer.Write(x);
        _writer.Write(z);
        _writer.Write(speedInstant);
        _writer.Write(speedMax);
        _writer.Write(radius);
        MaybeFlush();
    }

    private static ushort GetOrDefineObjectId(string objectName)
    {
        if (_objectIds.TryGetValue(objectName, out var id))
            return id;

        id = _nextObjectId++;
        _objectIds[objectName] = id;

        byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(objectName);
        byte nameLen = (byte)Math.Min(nameBytes.Length, 255);

        _writer.Write((byte)0);       // tag: define object
        _writer.Write(id);
        _writer.Write(nameLen);
        _writer.Write(nameBytes, 0, nameLen);
        MaybeFlush();

        return id;
    }

    private static void MaybeFlush()
    {
        _writesSinceFlush++;
        if (_writesSinceFlush >= FlushEvery)
        {
            _writer.Flush();
            _writesSinceFlush = 0;
        }
    }
}
