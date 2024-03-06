import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
TrainTabContext, FolderLocationContext, InterrogatorNameContext, SocketContext, TrainStartedContext, TrainProgressContext,
TrainProgressTextContext, TrainCompletedContext, TrainImagesContext, ReverseSortContext} from "../Context"
import {ProgressBar} from "react-bootstrap"
import functions from "../structures/Functions"
import folder from "../assets/icons/folder.png"
import TrainImage from "../components/TrainImage"
import "./styles/traintag.less"
import axios from "axios"

let scrollLock = false

const TrainTag: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {trainTab, setTrainTab} = useContext(TrainTabContext)
    const {folderLocation, setFolderLocation} = useContext(FolderLocationContext)
    const {interrogatorName, setInterrogatorName} = useContext(InterrogatorNameContext)
    const {trainImages, setTrainImages} = useContext(TrainImagesContext)
    const {trainProgress, setTrainProgress} = useContext(TrainProgressContext)
    const {trainProgressText, setTrainProgressText} = useContext(TrainProgressTextContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {reverseSort, setReverseSort} = useContext(ReverseSortContext)
    const [append, setAppend] = useState("")
    const [slice, setSlice] = useState([])
    const [sliceIndex, setSliceIndex] = useState(0)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const max = 100 + (sliceIndex * 100)
        let slice = reverseSort ? trainImages.slice(Math.max(trainImages.length - max - 1, 0), trainImages.length - 1) : trainImages.slice(0, max)
        setSlice(slice)
    }, [trainImages, reverseSort, sliceIndex])

    const handleScroll = (event: Event) => {
        if(!slice.length) return
        if (scrollLock) return
        if (Math.abs(document.body.scrollHeight - (document.body.scrollTop + document.body.clientHeight)) <= 1) {
            scrollLock = true
            setSliceIndex((prev: number) => prev + 1)
            setTimeout(() => {
                scrollLock = false
            }, 1000)
        }
    }

    useEffect(() => {
        document.body.addEventListener("scroll", handleScroll)
        return () => {
            document.body.removeEventListener("scroll", handleScroll)
        }
    }, [slice])

    const imagesJSX = () => {
        let jsx = [] as any
        if (reverseSort) {
            for (let i = slice.length - 1; i >= 0; i--) {
                jsx.push(<TrainImage img={trainImages[i]}/>)
            }
        } else {
            for (let i = 0; i < slice.length; i++) {
                jsx.push(<TrainImage img={trainImages[i]}/>)
            }
        }
        return jsx
    }

    useEffect(() => {
        const savedAppend = localStorage.getItem("append")
        if (savedAppend) setAppend(savedAppend)
    }, [])

    useEffect(() => {
        localStorage.setItem("append", String(append))
    }, [append])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(-1)
            setTrainProgressText("")
        }
        const trackProgress = (data: any) => {
            const progress = (100 / Number(data.total_step)) * Number(data.step)
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(progress)
            setTrainProgressText(`${data.step} / ${data.total_step}`)
        }
        const completeTrain = async (data: any) => {
            setTrainCompleted(true)
            setTrainStarted(false)
        }
        const interruptTrain = () => {
            setTrainStarted(false)
        }
        socket.on("train starting", startTrain)
        socket.on("train progress", trackProgress)
        socket.on("train complete", completeTrain)
        socket.on("train interrupt", interruptTrain)
        return () => {
            socket.off("train starting", startTrain)
            socket.off("train progress", trackProgress)
            socket.off("train complete", completeTrain)
            socket.off("train interrupt", interruptTrain)
        }
    }, [socket])

    const updateLocation = async () => {
        const location = await axios.post("/update-location").then((r) => r.data)
        if (location) setFolderLocation(location)
    }

    useEffect(() => {
        const updateTrainImages = async () => {
            let images = await axios.post("/list-files", {folder: folderLocation}).then((r) => r.data)
            if (images?.length) {
                images = images.map((i: string) => `/retrieve?path=${i}&?v=${new Date().getTime()}`)
                setTrainImages(images)
            }
        }
        updateTrainImages()
    }, [folderLocation])

    const getText = () => {
        if (trainCompleted) return "Completed"
        if (trainProgress >= 0) return trainProgressText
        return "Starting"
    }

    const getProgress = () => {
        if (trainCompleted) return 100
        if (trainProgress >= 0) return trainProgress
        return 0
    }

    const openImageLocation = async () => {
        await axios.post("/open-folder", {absolute: folderLocation})
    }

    const tag = async () => {
        await axios.post("/tag", {images: trainImages.map((i: string) => i.replace("/retrieve?path=", "").split("&")[0]), model: interrogatorName, append})
    }

    const interruptTag = async () => {
        axios.post("/interrupt-train")
    }

    const deleteTags = async () => {
        await axios.post("/delete-tags", {images: trainImages.map((i: string) => i.replace("/retrieve?path=", "").split("&")[0])})
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-folder-container">
                <img className="train-tag-folder" src={folder} style={{filter: getFilter()}} onClick={updateLocation}/>
                <div className="train-tag-location" onDoubleClick={openImageLocation}>{folderLocation ? folderLocation : "None"}</div>
                <button className="train-tag-button" onClick={() => trainStarted ? interruptTag() : tag()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{trainStarted ? "Stop" : "Tag"}</button>
                <button className="train-tag-button" onClick={() => deleteTags()}>Delete Tags</button>
            </div>
            <div className="train-tag-settings-container">
                <div className="train-tag-settings-column">
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Append:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={append} onChange={(event) => setAppend(event.target.value)}/>
                    </div>
                </div>
            </div>
            {trainStarted ? <div className="train-tag-progress">
                <div className="render-progress-container" style={{filter: getFilter()}}>
                    <span className="render-progress-text">{getText()}</span>
                    <ProgressBar ref={progressBarRef} animated now={getProgress()}/>
                </div>
            </div> : null}
            <div className="train-tag-images-container">
                {imagesJSX()}
            </div>
        </div>
    )
}

export default TrainTag