#ifndef CAFFE_MESSENGER_H_
#define CAFFE_MESSENGER_H_

#include <map>
#include <list>
#include <vector>
#include <string>
#include "caffe/common.hpp"

namespace caffe {

class Messenger {
private:
  static inline int& getMessengerIter() {
    static int messenger_iter;
    return messenger_iter;
  }

protected:
  static inline bool& getAsyncReadFlag() {
    static bool async_read_flag = false;
    return async_read_flag;
  }

  static inline bool chkAsyncReadFlag() {
    if (getAsyncReadFlag() == false) {
      getAsyncReadFlag() = true;
      return false;
    }
    return true;
  }

  static inline const int& GetIter() {
    return getMessengerIter();
  }

public:
  static inline void SetIter(int iter) {
    getMessengerIter() = iter;
    getAsyncReadFlag() = false;
  }
};

class AsyncMessenger : public Messenger {
private:
  typedef struct {
    int interval;
    int postpone;
    int duration;
    int livecost;
    const void* message;
  } DetailBlk;
  class CommndMap : public std::map<string, DetailBlk> {};
  class TargetMap : public std::map<string, CommndMap> {};
  class SourceMap : public std::map<string, TargetMap> {};
  class PullerMap : public std::map<string, SourceMap> {};
  class PusherQry : public std::map<string, PullerMap> {};

  AsyncMessenger() {} // AsyncMessenger should never be instantiated.

  static PusherQry& GetPusherQry() {
    static PusherQry* p_pusher_qry_ = new PusherQry();
    return *p_pusher_qry_;
  }

  static bool HasMessage(
      const string& pusher, const string& puller, const string& source,
      const string& target, const string& command) {
    const PusherQry& pusher_qry = GetPusherQry();
    typename PusherQry::const_iterator pusher_itr = pusher_qry.find(pusher);
    if (pusher_itr == pusher_qry.end()) return false;
    const PullerMap& puller_map = pusher_itr->second;
    typename PullerMap::const_iterator puller_itr = puller_map.find(puller);
    if (puller_itr == puller_map.end()) return false;
    const SourceMap& source_map = puller_itr->second;
    typename SourceMap::const_iterator source_itr = source_map.find(source);
    if (source_itr == source_map.end()) return false;
    const TargetMap& target_map = source_itr->second;
    typename TargetMap::const_iterator target_itr = target_map.find(target);
    if (target_itr == target_map.end()) return false;
    const CommndMap& commnd_map = target_itr->second;
    typename CommndMap::const_iterator commnd_itr = commnd_map.find(command);
    if (commnd_itr == commnd_map.end()) return false;
    return true;
  }

public:
  static void ClearMessages() {
    PusherQry& pusher_qry = GetPusherQry();
    pusher_qry.clear();
  }

  static bool PushMessage(
      const string& pusher, const string& puller, const string& source, const string& target, const string& command,
      const int interval,   const int postpone,   const int duration,   const int livetime,   const void* message = 0) {
    if (!chkAsyncReadFlag()) ClearMessages();
    const int iter = Messenger::GetIter();
    if (iter % interval < postpone || iter % interval >= postpone + duration) return false;
    CHECK(!HasMessage(pusher, puller, source, target, command))
        << "The duplicated message exists in messenger.";
    PusherQry& pusher_qry = GetPusherQry();
    DetailBlk& detail_blk = pusher_qry[pusher][puller][source][target][command];
    detail_blk.interval = interval;
    detail_blk.postpone = postpone;
    detail_blk.duration = duration;
    detail_blk.livecost = livetime;
    detail_blk.message  = message;
    return true;
  }

  static bool PullMessage(
      const string& pusher, const string& puller, const string& source, const string& target, const string& command,
      const int interval,   const int postpone,   const int duration,   const int costtime,   const void **message = 0) {
    *message = NULL;
    const int iter = Messenger::GetIter();
    if (iter % interval < postpone || iter % interval >= postpone + duration) return false;
    PusherQry& pusher_qry = GetPusherQry();
    typename PusherQry::iterator pusher_itr = pusher_qry.find(pusher);
    if (pusher_itr == pusher_qry.end()) return false;
    PullerMap& puller_map = pusher_itr->second;
    typename PullerMap::iterator puller_itr = puller_map.find(puller);
    if (puller_itr == puller_map.end()) return false;
    SourceMap& source_map = puller_itr->second;
    typename SourceMap::iterator source_itr = source_map.find(source);
    if (source_itr == source_map.end()) return false;
    TargetMap& target_map = source_itr->second;
    typename TargetMap::iterator target_itr = target_map.find(target);
    if (target_itr == target_map.end()) return false;
    CommndMap& commnd_map = target_itr->second;
    typename CommndMap::iterator commnd_itr = commnd_map.find(command);
    if (commnd_itr == commnd_map.end()) return false;
    DetailBlk& detail_blk = commnd_itr->second;
    if (detail_blk.livecost < costtime) return false;
    *message = detail_blk.message;
    if (detail_blk.livecost > costtime) {
      detail_blk.livecost -= costtime;
      return true;
    }
    commnd_map.erase(commnd_itr);
    if (!commnd_map.empty()) return true;
    target_map.erase(target_itr);
    if (!target_map.empty()) return true;
    source_map.erase(source_itr);
    if (!source_map.empty()) return true;
    puller_map.erase(puller_itr);
    if (!puller_map.empty()) return true;
    pusher_qry.erase(pusher_itr);
    return true;
  }
};
// class AsyncMessenger

class Listener {
public:
  virtual ~Listener() {}
  virtual void handle(const void* message) = 0;
};
// class Listener

class SyncMessenger : public Messenger {
private:
  typedef struct {
    int interval;
    int postpone;
    int duration;
    Listener* listener;
  } ListenBlk;
  typedef std::list<ListenBlk> ListenLst;
  class CommndMap : public std::map<string, ListenLst> {};
  class TargetMap : public std::map<string, CommndMap> {};
  class SourceMap : public std::map<string, TargetMap> {};
  class PullerMap : public std::map<string, SourceMap> {};
  class PusherQry : public std::map<string, PullerMap> {};

  SyncMessenger() {} // SyncMessenger should never be instantiated.

  static PusherQry& GetPusherQry() {
    static PusherQry* p_pusher_qry_ = new PusherQry();
    return *p_pusher_qry_;
  }

public:
  static void ClearListeners() {
    PusherQry& pusher_qry = GetPusherQry();
    pusher_qry.clear();
  }

  static void AddListener(
      const string& pusher, const string& puller, const string& source, const string& target, const string& commnd, 
      const int interval,   const int postpone,   const int duration,   Listener* listener) {
    ListenBlk listen_blk;
    listen_blk.interval = interval;
    listen_blk.postpone = postpone;
    listen_blk.duration = duration;
    listen_blk.listener = listener;
    PusherQry& pusher_qry = GetPusherQry();
    pusher_qry[pusher][puller][source][target][commnd].push_back(listen_blk);
  }

  static bool PushMessage(
      const string& pusher, const string& puller, const string& source, const string& target, const string& commnd,
      const int interval,   const int postpone,   const int duration,   const void* message) {
    const int iter = Messenger::GetIter();
    if (iter % interval < postpone || iter % interval >= postpone + duration) return false;
    PusherQry& pusher_qry = GetPusherQry();
    typename PusherQry::const_iterator pusher_itr = pusher_qry.find(pusher);
    if (pusher_itr == pusher_qry.end()) return false;
    const PullerMap& puller_map = pusher_itr->second;
    typename PullerMap::const_iterator puller_itr = puller_map.find(puller);
    if (puller_itr == puller_map.end()) return false;
    const SourceMap& source_map = puller_itr->second;
    typename SourceMap::const_iterator source_itr = source_map.find(source);
    if (source_itr == source_map.end()) return false;
    const TargetMap& target_map = source_itr->second;
    typename TargetMap::const_iterator target_itr = target_map.find(target);
    if (target_itr == target_map.end()) return false;
    const CommndMap& commnd_map = target_itr->second;
    typename CommndMap::const_iterator commnd_itr = commnd_map.find(commnd);
    if (commnd_itr == commnd_map.end()) return false;
    const ListenLst& listen_lst = commnd_itr->second;
    for (ListenLst::const_iterator listen_itr = listen_lst.begin(); listen_itr != listen_lst.end(); ++listen_itr) {
      const ListenBlk& listen_blk = *listen_itr;
      if (iter % listen_blk.interval >= listen_blk.postpone && iter % listen_blk.interval < listen_blk.postpone + listen_blk.duration) {
        listen_blk.listener->handle(message);
      }
    }
    return true;
  }
};
// class SyncMessenger

}  // namespace caffe

#endif  // CAFFE_MESSENGER_H_