import yt_dlp
import logging
import os

logger = logging.getLogger(__name__)

class YouTubeProcessor:
    def __init__(self):
        """
        YouTube 처리기 초기화.
        """
        # 기본 yt-dlp 옵션
        self.base_ydl_opts = {
            'format': 'bestaudio/best', 
            'postprocessors': [{
                'key': 'FFmpegExtractAudio', 
                'preferredcodec': 'wav',     
                # 'preferredquality': '192', # 오디오 품질 (선택 사항)
            }],
            'noplaylist': True,          
            'quiet': True,               
            'noprogress': True,          
            'socket_timeout': 30,        
            # 'verbose': True, # 디버깅 시 유용
        }
        logger.info("YouTubeProcessor가 초기화되었습니다.")

    def download_audio(self, url: str, output_dir: str = None, filename_pattern: str = '%(id)s.%(ext)s') -> str | None:
        logger.info(f"YouTube 오디오 다운로드 요청: {url}, 출력 디렉토리: {output_dir}")

        current_ydl_opts = self.base_ydl_opts.copy()
        
        if output_dir:
            # 출력 템플릿을 설정하여 지정된 디렉토리에 저장
            current_ydl_opts['outtmpl'] = os.path.join(output_dir, filename_pattern)
        else:
            # output_dir이 없으면 기본 outtmpl 사용 (현재 디렉토리에 id.ext 형태로 저장)
            current_ydl_opts['outtmpl'] = filename_pattern


        try:
            with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
                logger.debug(f"yt-dlp 인스턴스 생성 및 정보 추출 시도: {url}")
                info = ydl.extract_info(url, download=True) # download=True로 실제 다운로드 실행
                downloaded_file_path = ydl.prepare_filename(info)
                base, _ = os.path.splitext(downloaded_file_path)
                final_wav_path = base + ".wav"

                if os.path.exists(final_wav_path):
                    logger.info(f"오디오 다운로드 및 변환 성공: {final_wav_path}")
                    return final_wav_path
                else:
                    expected_path_from_outtmpl = current_ydl_opts['outtmpl'] \
                        .replace('%(id)s', info.get('id', 'unknown_id')) \
                        .replace('%(ext)s', info.get('ext', 'mp4')) # 원본 ext
                    
                    base_expected, _ = os.path.splitext(expected_path_from_outtmpl)
                    final_wav_path_expected = base_expected + ".wav"

                    if os.path.exists(final_wav_path_expected):
                         logger.info(f"오디오 다운로드 및 변환 성공 (예상 경로): {final_wav_path_expected}")
                         return final_wav_path_expected
                    else:
                        logger.error(f"다운로드된 WAV 파일을 찾을 수 없습니다. 예상 경로: {final_wav_path} 또는 {final_wav_path_expected}")
                        return None

        except yt_dlp.utils.DownloadError as e: # type: ignore
            logger.error(f"YouTube 다운로드 오류 ({url}): {e}", exc_info=True)
            # 특정 오류 유형에 따라 다른 처리 가능 (예: GeoRestrictedError, MaxDownloadsReached)
            return None
        except Exception as e:
            logger.error(f"YouTube 처리 중 예기치 않은 오류 ({url}): {e}", exc_info=True)
            return None
    def extract_video_id(self,url: str) -> str:
        """
        YouTube URL에서 비디오 ID를 추출합니다.

        Args:
            url (str): YouTube 동영상 URL

        Returns:
            str: 추출된 비디오 ID 또는 URL 파싱 실패 시 빈 문자열
        """
        try:
            # URL에서 v 파라미터 추출 시도
            if "youtube.com" in url and "v=" in url:
                query_str = url.split("?")[1]
                params = {p.split("=")[0]: p.split("=")[1] for p in query_str.split("&") if "=" in p}
                video_id = params.get("v", "")
            # youtu.be 형식인 경우
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = ""

            return video_id
        except Exception as e:
            logger.error(f"비디오 ID 추출 중 오류 발생: {e}")
            return ""

    def get_video_info(self,url: str,cookies_file: str = None, use_browser_cookies: str = None) -> dict:
        """
        YouTube 동영상 정보
        Args:
            url (str): YouTube 동영상 URL
        Returns:
            dict: 동영상 정보 (제목, ID, 등)
        """
        try:
            info_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': True,
            }

            # 쿠키 설정 추가
            if cookies_file:
                info_opts['cookiefile'] = cookies_file
            elif use_browser_cookies:
                info_opts['cookies_from_browser'] = use_browser_cookies

            # User-Agent 설정 추가 (선택적)
            info_opts['http_headers'] = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            }

            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # 비디오 정보가 없는 경우 확인
                if not info:
                    logger.warning(f"비디오 정보를 가져올 수 없습니다: {url}")
                    return {}

                return {
                    'id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'channel': info.get('channel', ''),
                    'channel_id': info.get('channel_id', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"비디오 정보 가져오기 실패: {e}", exc_info=True)
            # 단순히 빈 딕셔너리 대신 오류 메시지와 함께 반환
            return {
                'error': str(e),
                'id': self.extract_video_id(url) or '',
            }